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

# winget id per tool. ezwinports.make rather than GnuWin32.Make: GnuWin32's is make 3.81 from
# 2006 and is not put on PATH, this one is current and is.
$Packages = [ordered]@{
  git  = 'Git.Git'
  node = 'OpenJS.NodeJS.LTS'
  uv   = 'astral-sh.uv'
  make = 'ezwinports.make'
}

function Say([string]$m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "  ok      $m" -ForegroundColor Green }
function Miss([string]$m) { Write-Host "  missing $m" -ForegroundColor Yellow }
function Have([string]$c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

# winget changes PATH in the registry only; pull it into this session so the tools it just
# installed are callable without opening a new terminal.
function Update-SessionPath {
  $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  $user = [Environment]::GetEnvironmentVariable('Path', 'User')
  $env:Path = "$machine;$user;$env:LOCALAPPDATA\Microsoft\WinGet\Links;$env:USERPROFILE\.local\bin"
}

function Get-NodeMajor {
  if (-not (Have node)) { return 0 }
  try { return [int](& node -p "process.versions.node.split('.')[0]") } catch { return 0 }
}

function Invoke-Checked([string]$what, [scriptblock]$block) {
  & $block
  if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
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

if ($Check) {
  if ((Have uv) -and ((& uv python find $PythonVersion 2>$null) -ne $null)) {
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
    & winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements
    # -1978335189 = already installed; winget says so when the tool is installed but not on PATH yet.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
      throw "winget install $id failed (exit $LASTEXITCODE)"
    }
  }
  Update-SessionPath
  foreach ($tool in $missing) {
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

Say 'Done. Next, from Git Bash in the repo folder:'
Write-Host '    make test   # confirm both toolchains work (no camera needed)'
Write-Host '    make dev    # launch the app'
