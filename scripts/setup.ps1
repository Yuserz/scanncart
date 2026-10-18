<#
.SYNOPSIS
  SCANnCART one-command setup for Windows.

.DESCRIPTION
  Installs the system tools this repo needs (the list below) with winget, then installs both
  toolchains - the desktop's npm packages and the sidecar's Python 3.12 venv - exactly as
  `make install` would. Re-running it is safe: a tool that is already there at a good enough
  version is left alone.

  The tools, and why (README.md "Prerequisites" is the human version of this list):
    Git (+ Git Bash)  any           clone the repo; Git Bash is the POSIX shell `make` needs
    Node.js + npm     20+           the Electron/React desktop app (CI uses 20)
    uv                any           creates the sidecar venv and fetches Python 3.12 itself
    Python            exactly 3.12  the sidecar - installed *by uv*, never the system python
    GNU Make          any           the root Makefile that wraps both toolchains
    VC++ runtime      2015-2022 x64 PyTorch's DLLs load against it; without it the sidecar dies on
                                    import with "WinError 126 ... fbgemm.dll"

  Linux/macOS: use scripts/setup.sh instead.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Check
#>
[CmdletBinding()]
param(
  # Install the system tools only, not the project's npm/pip dependencies.
  [switch]$ToolsOnly,
  # Report what is missing and change nothing.
  [switch]$Check,
  # Run both test suites at the end.
  [switch]$Test
)

$ErrorActionPreference = 'Stop'

$NodeMajorMin = 20
$PythonVersion = '3.12'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$OnWindows = $env:OS -eq 'Windows_NT'

# winget id per tool. ezwinports.make rather than GnuWin32.Make: GnuWin32's is make 3.81 from
# 2006 and is not put on PATH, this one is current and is.
$Packages = [ordered]@{
  git  = 'Git.Git'
  node = 'OpenJS.NodeJS.LTS'
  uv   = 'astral-sh.uv'
  make = 'ezwinports.make'
  vcredist = 'Microsoft.VCRedist.2015+.x64'
}

# What winget returns for "nothing to do": the package is already installed (no applicable upgrade
# / already installed). Both happen when a tool is installed but not on this session's PATH yet.
$WingetAlreadyInstalled = @(-1978335189, -1978335135)

function Say([string]$m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "  ok      $m" -ForegroundColor Green }
function Miss([string]$m) { Write-Host "  missing $m" -ForegroundColor Yellow }
function Have([string]$c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

# winget changes PATH in the registry only; pull it into this session so the tools it just
# installed are callable without opening a new terminal.
function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $user = [Environment]::GetEnvironmentVariable('Path', 'User')
  $extra = @($machine, $user, "$env:LOCALAPPDATA\Microsoft\WinGet\Links", "$env:USERPROFILE\.local\bin")
  $sep = [IO.Path]::PathSeparator
  $env:Path = (($extra + $env:Path) | Where-Object { $_ }) -join $sep
}

function Get-NodeMajor {
  if (-not (Have node)) { return 0 }
  try { return [int](& node -p "process.versions.node.split('.')[0]") } catch { return 0 }
}

# Native tools (uv, npm, winget) write progress to stderr. Windows PowerShell 5.1 turns those lines
# into error records when output is redirected, and under 'Stop' the first one aborts the script, so
# native calls run under 'Continue' and are judged by their exit code alone.
function Invoke-Checked([string]$what, [scriptblock]$block) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $block } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

function Test-Native([scriptblock]$block) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $block *> $null; return ($LASTEXITCODE -eq 0) } catch { return $false } finally { $ErrorActionPreference = $prev }
}

function Test-VCRedist {
  $key = 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
  try { return ((Get-ItemProperty $key -ErrorAction Stop).Installed -eq 1) } catch { return $false }
}

Update-SessionPath

# --- what is there ---------------------------------------------------------------------------

Say 'Checking system tools'
$missing = @()
if (Have git) { Ok (& git --version) } else { Miss 'git'; $missing += 'git' }
if ((Get-NodeMajor) -ge $NodeMajorMin -and (Have npm)) {
  Ok "node $(& node --version), npm $(& npm --version)"
} else {
  if (Have node) { Miss "node $NodeMajorMin+ (found $(& node --version))" } else { Miss "node $NodeMajorMin+" }
  $missing += 'node'
}
if (Have uv) { Ok (& uv --version) } else { Miss 'uv'; $missing += 'uv' }
if (Have make) { Ok ((& make --version) | Select-Object -First 1) } else { Miss 'make'; $missing += 'make' }
if ($OnWindows) {
  if (Test-VCRedist) { Ok 'Visual C++ runtime (x64)' } else { Miss 'Visual C++ runtime (x64)'; $missing += 'vcredist' }
}

if ($Check) {
  if ((Have uv) -and (Test-Native { & uv python find $PythonVersion })) {
    Ok "python $PythonVersion (via uv)"
  } else {
    Miss "python $PythonVersion (uv fetches it during setup)"
  }
  if (Test-Path "$RepoRoot\sidecar\.venv\Scripts\python.exe") { Ok 'sidecar\.venv' } else { Miss 'sidecar\.venv' }
  if (Test-Path "$RepoRoot\desktop\node_modules") { Ok 'desktop\node_modules' } else { Miss 'desktop\node_modules' }
  exit 0
}

# --- install what is missing -----------------------------------------------------------------

if ($missing.Count -gt 0) {
  if (-not (Have winget)) {
    throw 'winget is not available. Install "App Installer" from the Microsoft Store, then re-run.'
  }
  foreach ($tool in $missing) {
    $id = $Packages[$tool]
    Say "Installing $tool ($id)"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      & winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements
    } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0 -and $WingetAlreadyInstalled -notcontains $LASTEXITCODE) {
      throw "winget install $id failed (exit $LASTEXITCODE)"
    }
  }
  Update-SessionPath
  foreach ($tool in $missing) {
    if ($tool -eq 'vcredist') { continue }
    if (-not (Have $tool)) {
      throw "$tool was installed but is not on PATH yet - open a new terminal and re-run this script."
    }
  }
  if ((Get-NodeMajor) -lt $NodeMajorMin) { throw "Node.js $NodeMajorMin+ is required (found $(& node --version))." }
}

Say "Installing Python $PythonVersion (managed by uv)"
Invoke-Checked 'uv python install' { & uv python install $PythonVersion }

if ($ToolsOnly) { Say 'System tools ready'; exit 0 }

# --- project install (what `make install` does) ----------------------------------------------

Say "Installing desktop dependencies (npm install - Electron's download can take a while)"
Push-Location "$RepoRoot\desktop"
try { Invoke-Checked 'npm install' { & npm install } } finally { Pop-Location }

Say "Creating sidecar\.venv with Python $PythonVersion and installing requirements.txt"
Push-Location "$RepoRoot\sidecar"
try {
  if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Invoke-Checked 'uv venv' { & uv venv --python $PythonVersion .venv }
  }
  Invoke-Checked 'uv pip install' { & uv pip install --python .venv\Scripts\python.exe -r requirements.txt }
} finally { Pop-Location }

if ($Test) {
  # The same two suites `make test` runs, called directly so this works from PowerShell, where
  # make has no POSIX shell to run its recipes in.
  Say 'Running both test suites (fakes only - no camera needed)'
  Push-Location "$RepoRoot\desktop"
  try { Invoke-Checked 'npm test' { & npm test } } finally { Pop-Location }
  Push-Location "$RepoRoot\sidecar"
  try { Invoke-Checked 'pytest' { & .venv\Scripts\python.exe -m pytest -q } } finally { Pop-Location }
}

# npm, npx and uv ship .ps1 shims, and PowerShell's default policy ("Restricted") refuses to run
# them: the next `npm run dev` typed into PowerShell fails with "running scripts is disabled on
# this system". This script runs under -ExecutionPolicy Bypass so it is not affected; say how to
# fix the user's own shells rather than changing a security setting behind their back.
if ($OnWindows -and @('Restricted', 'AllSigned', 'Undefined') -contains [string](Get-ExecutionPolicy)) {
  Write-Host ''
  Write-Host "PowerShell's execution policy is '$(Get-ExecutionPolicy)', which blocks npm in PowerShell." -ForegroundColor Yellow
  Write-Host 'Allow local scripts for your user (once), or use Git Bash / cmd instead:' -ForegroundColor Yellow
  Write-Host '    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned'
}

Write-Host ''
Say 'Done. Next, from Git Bash in the repo folder (make needs a POSIX shell):'
Write-Host '    make test   # confirm both toolchains work (no camera needed)'
Write-Host '    make dev    # launch the app'
Write-Host 'Or without make, from any terminal:'
Write-Host '    cd desktop; npm run dev'
