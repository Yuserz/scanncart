<#
.SYNOPSIS
  SCANnCART one-command setup for Windows.

.DESCRIPTION
  Installs the system tools this repo needs (the list below) with winget, then installs both
  toolchains - the desktop's npm packages and the sidecar's Python 3.12 venv - exactly as
  `make install` would. Re-running it is safe: a tool that is already there at a good enough
  version is left alone.

  Every tool this repo uses, and why (README.md "Prerequisites" is the human version of this list;
  scripts/setup.sh carries the same list - change both together):

  Installed by this script:
    Node.js + npm     20+           required - the Electron/React desktop app (CI uses 20)
    uv                any           required - creates the sidecar venv and fetches Python 3.12
    Python            exactly 3.12  required - the sidecar; installed *by uv*, never the system one
    VC++ runtime      2015-2022 x64 required - PyTorch's DLLs load against it; without it the
                                    sidecar dies on import with "WinError 126 ... fbgemm.dll"
    Python packages   requirements.lock.txt  required - the exact versions, into sidecar\.venv
    npm packages      package-lock      required - into desktop\node_modules (Electron, React, vite...)
    Git               any           recommended - updates the checkout, and its sh.exe is the POSIX
                                    shell the Makefile's recipes run in (from any terminal)
    GNU Make          any           optional - the root Makefile; start.cmd launches without it

  Installed when an NVIDIA driver is (nvidia-smi on PATH):
    CUDA torch + onnxruntime-gpu    NVIDIA acceleration (docs/DEVELOPMENT.md section 6). Automatic:
                                    without it a GPU laptop runs on its CPU and the Admin panel
                                    recommends a CPU preset for it. -Cpu keeps the CPU builds;
                                    -Gpu insists, and fails when no NVIDIA driver is installed.

  Not installed by this script (optional, see the docs):
    NVIDIA driver     for the GPU   from nvidia.com - a driver is tied to the card, not the repo
    Docker Desktop    local_api     only for the Roboflow inference-server backend
                                    (docs/DETECTOR_BACKENDS.md); the default `native` backend
                                    needs nothing beyond the list above
    A camera          live capture  a USB webcam or Logitech StreamCam; tests need none

  The installed model weights are committed under sidecar\models, so a clone or `git pull` has
  them. Any the checkout lacks are downloaded from the repo's `models` GitHub Release
  (scripts/fetch_models.py; -NoModels skips it), and -ModelsFrom copies them from a folder instead
  (a USB stick, a shared drive). Neither replaces a weight already in sidecar\models. The default
  model is scanncart-grocery-v2.1.

  Linux/macOS: use scripts/setup.sh instead.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Check
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -ModelsFrom E:\scanncart-models
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu

  A full install ends by putting a "SCANnCART" shortcut to start.cmd on the desktop (-NoShortcut
  skips it; CI never gets one), so a new device launches from there.
#>
[CmdletBinding()]
param(
  # Install the system tools only, not the project's npm/pip dependencies.
  [switch]$ToolsOnly,
  # Report what is missing and change nothing.
  [switch]$Check,
  # Run both test suites at the end.
  [switch]$Test,
  # A folder holding model weights (*.pt + their *.json records) to copy into sidecar\models.
  [string]$ModelsFrom,
  # Skip downloading the weights from the repo's `models` GitHub Release.
  [switch]$NoModels,
  # Require the NVIDIA CUDA builds of torch/onnxruntime: fail when there is no NVIDIA driver.
  # (They are installed anyway whenever a driver is found; this only makes a missing one an error.)
  [switch]$Gpu,
  # Keep the CPU builds even on a machine with an NVIDIA driver.
  [switch]$Cpu,
  # Override the PyTorch CUDA wheel tag. Empty (the default) lets scripts/cuda_torch.py choose from
  # the card: cu128 for Turing (RTX 20xx/GTX 16xx) and newer, cu126 for older cards.
  [string]$CudaTag = '',
  # Skip the "SCANnCART" shortcut to start.cmd on the desktop.
  [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'
# uv gives up on a request after 30 s by default; torch's wheels are ~2.5 GB and shop/school Wi-Fi
# stalls longer than that. A user's own value wins.
if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '120' }
if ($Gpu -and $Cpu) { throw '-Gpu and -Cpu contradict each other; pass one.' }

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

# The app runs without these; a failure to install one is a warning, not the end of the setup.
$Optional = @('git', 'make')

# Where to get each tool by hand when neither winget nor a fallback can install it.
$ManualUrl = @{
  git  = 'https://git-scm.com/download/win'
  node = 'https://nodejs.org/en/download'
  uv   = 'https://docs.astral.sh/uv/getting-started/installation/'
  make = 'https://github.com/mbuilov/gnumake-windows'
  vcredist = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
}

# What winget returns for "nothing to do": the package is already installed (no applicable upgrade
# / already installed). Both happen when a tool is installed but not on this session's PATH yet.
$WingetAlreadyInstalled = @(-1978335189, -1978335135)

function Say([string]$m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok([string]$m)   { Write-Host "  ok      $m" -ForegroundColor Green }
function Miss([string]$m) { Write-Host "  missing $m" -ForegroundColor Yellow }
function Warn([string]$m) { Write-Host "warning: $m" -ForegroundColor Yellow }
function Have([string]$c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

# winget changes PATH in the registry only; pull it into this session so the tools it just
# installed are callable without opening a new terminal. The session's own PATH stays first (the
# node it already runs keeps winning) and a registry entry is added only when it is not already
# there: the session PATH already carries most of the registry's, and simply joining the two
# doubles it - long enough that the cmd.exe npm starts for install scripts cannot search it, and
# `node` "is not recognized" mid-install (seen on the Windows CI runner).
function Update-SessionPath {
  $sep = [IO.Path]::PathSeparator
  # The two WinGet\Links folders are where winget puts portable tools (uv, make): per user, or
  # machine-wide when winget ran elevated. ~\.local\bin is where uv's own installer puts uv.
  $sources = @($env:Path,
    [Environment]::GetEnvironmentVariable('Path', 'Machine'),
    [Environment]::GetEnvironmentVariable('Path', 'User'),
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links", "$env:ProgramFiles\WinGet\Links",
    "$env:USERPROFILE\.local\bin")
  $seen = @{}
  $entries = New-Object System.Collections.Generic.List[string]
  foreach ($source in $sources) {
    if (-not $source) { continue }
    foreach ($entry in $source.Split($sep)) {
      $entry = $entry.Trim()
      $key = $entry.TrimEnd('\', '/').ToLowerInvariant()
      if ($key -and -not $seen.ContainsKey($key)) { $seen[$key] = $true; $entries.Add($entry) }
    }
  }
  $env:Path = $entries -join $sep
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

# Where the Makefile looks for Git's sh.exe (its GIT_SH list): without one, make runs its recipes
# in cmd.exe and they fail, so a make with no Git beside it is worth a warning.
function Get-GitShell {
  $candidates = @("$env:ProgramFiles\Git\bin\sh.exe", "${env:ProgramFiles(x86)}\Git\bin\sh.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\sh.exe")
  return ($candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1)
}

# The policy the user's own PowerShell windows get. Get-ExecutionPolicy alone would answer for this
# process, which setup.cmd starts with -ExecutionPolicy Bypass, so it would never report a problem.
# The first scope that sets one wins (group policy first); none set means Windows' default.
function Get-UserShellPolicy {
  foreach ($scope in 'MachinePolicy', 'UserPolicy', 'CurrentUser', 'LocalMachine') {
    $policy = [string](Get-ExecutionPolicy -Scope $scope)
    if ($policy -ne 'Undefined') { return $policy }
  }
  return 'Restricted'
}

# Installs one tool, by winget when there is one, else by the tool's own installer where it has
# one that needs no winget (uv, the VC++ runtime). Returns $false instead of throwing, so the
# caller decides whether the tool was required.
function Install-Tool([string]$tool) {
  if (Have winget) {
    $id = $Packages[$tool]
    Say "Installing $tool ($id)"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      & winget install --id $id --exact --silent --accept-package-agreements --accept-source-agreements
    } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -eq 0 -or $WingetAlreadyInstalled -contains $LASTEXITCODE) { return $true }
    Warn "winget install $id failed (exit $LASTEXITCODE)"
  }
  try {
    switch ($tool) {
      'uv' {
        Say 'Installing uv with its own installer'
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        return $true
      }
      'vcredist' {
        Say 'Installing the Visual C++ runtime from Microsoft (asks for admin rights)'
        $exe = Join-Path $env:TEMP 'vc_redist.x64.exe'
        Invoke-WebRequest -UseBasicParsing -Uri $ManualUrl['vcredist'] -OutFile $exe
        $p = Start-Process -FilePath $exe -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -Wait -PassThru
        # 3010 = installed, a restart finishes it; 1638 = a newer one is already there.
        return (@(0, 3010, 1638) -contains $p.ExitCode)
      }
    }
  } catch { Warn "installing $tool failed: $($_.Exception.Message)" }
  return $false
}

Update-SessionPath

# A checkout under OneDrive is the most common way this setup goes wrong on a student's laptop:
# node_modules and the venv are tens of thousands of small files, OneDrive starts syncing every one,
# and installs then fail on files it holds open (EPERM / "being used by another process"), or crawl.
# Said first, before minutes of downloading, because the fix is to move the folder, not to retry.
if ($OnWindows -and $RepoRoot -match '\\OneDrive( - [^\\]+)?\\') {
  Warn "This folder is inside OneDrive ($RepoRoot)."
  Warn 'OneDrive will try to sync the ~50,000 files the install creates, which is slow and makes installs fail with'
  Warn '"file in use" / EPERM errors. Move the folder somewhere OneDrive does not sync - e.g. C:\dev\scanncart - and'
  Warn 'run setup.cmd from there. (Continuing anyway in 10 seconds; press Ctrl+C to stop.)'
  Start-Sleep -Seconds $(if ($env:CI) { 0 } else { 10 })
}

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
  if (Get-GitShell) { Ok "Git's sh.exe for make: $(Get-GitShell)" } else { Miss "Git's sh.exe (make's recipes need it)" }
}
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
  $weights = @(Get-ChildItem "$RepoRoot\sidecar\models" -Filter *.pt -ErrorAction SilentlyContinue)
  if ($weights.Count -gt 0) { Ok "model weights: $(($weights | ForEach-Object Name) -join ', ')" } else { Miss 'model weights in sidecar\models' }
  if (Have nvidia-smi) {
    $venvPy = "$RepoRoot\sidecar\.venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
      & $venvPy -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
      if ($LASTEXITCODE -eq 0) { Ok 'CUDA torch (the NVIDIA GPU runs inference)' } else { Miss 'CUDA torch: an NVIDIA driver is installed but torch runs on the CPU (re-run setup.cmd)' }
    }
  }
  if (Test-Path "$RepoRoot\sidecar\models\scanncart-grocery-v2.1.pt") { Ok 'default model scanncart-grocery-v2.1.pt' } else { Miss 'default model scanncart-grocery-v2.1.pt (committed in git; setup also downloads it from the models release)' }
  exit 0
}

# --- install what is missing -----------------------------------------------------------------

if ($missing.Count -gt 0) {
  if (-not (Have winget)) {
    Warn 'winget is not available (install "App Installer" from the Microsoft Store to get it).'
    Warn 'Falling back to the tools'' own installers where they have one.'
  }
  $failed = @()
  foreach ($tool in $missing) {
    if (-not (Install-Tool $tool)) { $failed += $tool }
  }
  Update-SessionPath
  foreach ($tool in $missing) {
    $present = if ($tool -eq 'vcredist') { Test-VCRedist } else { Have $tool }
    if (-not $present -and $failed -notcontains $tool) { $failed += $tool }
  }
  $blocking = @($failed | Where-Object { $Optional -notcontains $_ })
  foreach ($tool in $failed) {
    if ($Optional -contains $tool) { Warn "$tool is optional and was not installed - get it from $($ManualUrl[$tool])" }
  }
  if ($blocking.Count -gt 0) {
    foreach ($tool in $blocking) { Write-Host "  install $tool from $($ManualUrl[$tool])" -ForegroundColor Red }
    throw ("Required tools are missing: $($blocking -join ', '). Install them (links above), " +
      'open a new terminal and re-run. If one was just installed, a new terminal may be all it needs.')
  }
  if ((Get-NodeMajor) -lt $NodeMajorMin) { throw "Node.js $NodeMajorMin+ is required (found $(& node --version))." }
}

Say "Installing Python $PythonVersion (managed by uv)"
Invoke-Checked 'uv python install' { & uv python install $PythonVersion }

if ($ToolsOnly) { Say 'System tools ready'; exit 0 }

# --- project install (what `make install` does) ----------------------------------------------

# `npm ci`, not `npm install`: it installs exactly the lockfile and never writes it. npm 10 (Node
# 22's) rewrites a lockfile npm 11 wrote, so `npm install` left a fresh clone with a modified file.
# `npm ci` deletes node_modules first, and Windows refuses to delete a file a running program holds:
# an open (or orphaned) SCANnCART window keeps node_modules\electron\dist\debug.log open and npm
# fails with "EBUSY: resource busy or locked". Stop this checkout's own processes - the Electron
# app, its sidecar, vite/esbuild - before reinstalling. A process counts as this checkout's when its
# executable lives in the repo folder (electron.exe, esbuild.exe) or its command line names it (the
# dev server runs in the system node.exe, the sidecar in uv's python.exe behind the venv's
# launcher), and only these four program names are considered, so the shell running this script
# is never one of them. Each is ended with its children (taskkill /T): killing only the venv's
# launcher would leave the real interpreter running with the camera open.
function Stop-RepoProcesses {
  $names = @('electron.exe', 'node.exe', 'python.exe', 'esbuild.exe')
  $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.ProcessId -ne $PID -and $names -contains $_.Name.ToLower() -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith("$RepoRoot\", [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf("$RepoRoot\", [StringComparison]::OrdinalIgnoreCase) -ge 0))
    })
  foreach ($p in $procs) {
    Warn "Closing $($p.Name) (pid $($p.ProcessId)), run from this folder, so its files can be replaced"
    & taskkill.exe /PID $p.ProcessId /T /F 2>&1 | Out-Null
  }
  if ($procs.Count -gt 0) { Start-Sleep -Seconds 2 }
}

Say "Installing desktop dependencies (npm ci - Electron's download can take a while)"
Push-Location "$RepoRoot\desktop"
try {
  if ($OnWindows) { Stop-RepoProcesses }
  try {
    Invoke-Checked 'npm ci' { & npm ci }
  } catch {
    if (-not $OnWindows) { throw }
    # Something outside the repo can hold a file too (an editor, Explorer's preview, an antivirus
    # scan that finishes a moment later): retry once before giving up.
    Warn 'npm ci failed - closing anything still holding desktop\node_modules and trying once more'
    Stop-RepoProcesses
    Start-Sleep -Seconds 5
    try {
      Invoke-Checked 'npm ci' { & npm ci }
    } catch {
      # Keep npm's own failure first: a network or registry error must not read as a locked file.
      throw ("$($_.Exception.Message), twice - npm's own error is above. If it says EBUSY or EPERM, a " +
        'program still has a file in desktop\node_modules open: close SCANnCART, any editor or terminal ' +
        'opened in the desktop folder and Explorer windows showing it (or restart Windows), then run ' +
        'setup.cmd again.')
    }
  }
  # `npm ci` exits 0 even when Electron's own download/extract step produced nothing (seen with an
  # antivirus scanning the zip): node_modules\electron is there, electron.exe and path.txt are not,
  # and `npm run dev` later fails with the misleading "Electron uninstall". Check the binary itself,
  # re-run Electron's installer once, then once more from a fresh download cache.
  $electronOk = { Test-Native { & node -e "require('fs').accessSync(require('electron'))" } }
  if (-not (& $electronOk)) {
    Warn "Electron's binary is missing after npm ci - re-running its installer"
    Invoke-Checked "Electron's installer" { & node node_modules\electron\install.js }
    if (-not (& $electronOk)) {
      $env:electron_config_cache = Join-Path $env:TEMP "scanncart-electron-cache-$PID"
      Warn "Still missing - downloading Electron again into a fresh cache ($env:electron_config_cache)"
      Remove-Item -Recurse -Force node_modules\electron\dist -ErrorAction SilentlyContinue
      Invoke-Checked "Electron's installer" { & node node_modules\electron\install.js }
    }
    if (-not (& $electronOk)) {
      throw ("Electron did not install (desktop\node_modules\electron\dist has no electron.exe). An antivirus " +
        'quarantining it is the usual cause - allow the repo folder, delete desktop\node_modules and re-run.')
    }
  }
  Ok "Electron $(& node -p "require('electron/package.json').version")"
} finally { Pop-Location }

Say "Creating sidecar\.venv with Python $PythonVersion and installing requirements.lock.txt"
Push-Location "$RepoRoot\sidecar"
try {
  if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Invoke-Checked 'uv venv' { & uv venv --python $PythonVersion .venv }
  }
  # One retry: a single PyPI timeout on slow Wi-Fi failed the whole setup after minutes of
  # downloading. uv's cache keeps what already arrived, so the retry fetches only the rest.
  try {
    Invoke-Checked 'uv pip install' { & uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt }
  } catch {
    Warn "$($_.Exception.Message) - retrying once (a slow or dropped connection is the usual cause)"
    Invoke-Checked 'uv pip install' { & uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt }
  }

  # Prove the venv can load what the sidecar loads, now rather than at the first app start: a
  # missing VC++ runtime or a broken wheel fails here with its real error.
  Say 'Checking the sidecar can import its libraries'
  Invoke-Checked 'the import check' {
    & .venv\Scripts\python.exe -c "import torch, cv2, ultralytics, fastapi, onnxruntime, lap; print('  ok      torch', torch.__version__, '/ opencv', cv2.__version__, '/ ultralytics', ultralytics.__version__)"
  }

  # The NVIDIA builds of torch and onnxruntime, automatic whenever an NVIDIA driver is installed:
  # a plain install pulls CPU torch even on a CUDA machine, and the app then runs on the CPU and
  # recommends a CPU preset. scripts/cuda_torch.py is the one copy of the rule - `make sidecar-setup`
  # runs it too. -Cpu keeps the CPU builds; -Gpu makes a missing driver or unusable GPU an error.
  $cudaArgs = @()
  if ($CudaTag) { $cudaArgs += @('--cuda-tag', $CudaTag) }
  if ($Gpu) { $cudaArgs += '--require' }
  if ($Cpu) { $cudaArgs += '--cpu' }
  Invoke-Checked 'the GPU builds' { & .venv\Scripts\python.exe "$RepoRoot\scripts\cuda_torch.py" @cudaArgs }
} finally { Pop-Location }

# --- model weights (committed; the release fills any gap) --------------------------------------------------------------

$modelsDir = Join-Path $RepoRoot 'sidecar\models'
if ($ModelsFrom) {
  if (-not (Test-Path $ModelsFrom -PathType Container)) { throw "-ModelsFrom: '$ModelsFrom' is not a folder." }
  $copied = @(Get-ChildItem $ModelsFrom -File | Where-Object { @('.pt', '.json', '.onnx') -contains $_.Extension })
  if ($copied.Count -eq 0) { throw "-ModelsFrom: '$ModelsFrom' holds no .pt, .json or .onnx files." }
  Say "Copying $($copied.Count) model file(s) into sidecar\models"
  foreach ($f in $copied) {
    # Never replace a weight already here: a same-named file may be a different training run.
    if (Test-Path (Join-Path $modelsDir $f.Name)) { Warn "$($f.Name) is already in sidecar\models - kept, not replaced"; continue }
    Copy-Item $f.FullName -Destination $modelsDir
    Ok $f.Name
  }
}
# Then whatever the repo's `models` release has that this checkout does not (the committed weights
# are already here): scripts/fetch_models.py,
# the one downloader both setup scripts and `make models` share. It keeps any weight already here
# (including what -ModelsFrom just copied), and a missing release or network only warns.
if (-not $NoModels) {
  Say 'Downloading the model weights from the repo''s GitHub release'
  Push-Location $RepoRoot
  try { Invoke-Checked 'the weights download' { & sidecar\.venv\Scripts\python.exe scripts\fetch_models.py } } finally { Pop-Location }
}

# The app's default model (sidecar/app/settings.py). Its .json record is what tells `auto` the
# stretch geometry it was trained at, so the weight without its record is half an install.
$defaultModel = 'scanncart-grocery-v2.1'
if ($OnWindows -and (Have make) -and -not (Get-GitShell)) {
  Warn "make is installed but Git is not, so make's recipes have no POSIX shell. Install Git ($($ManualUrl['git']))."
}
if (-not (Test-Path (Join-Path $modelsDir "$defaultModel.pt"))) {
  Write-Host ''
  Warn "The default model, sidecar\models\$defaultModel.pt, is not here - the app starts, but has no grocery detector."
  Warn 'It is committed in git - run `git pull` - or attach it and its .json to the repo''s `models` release, copy them in with'
  Warn '-ModelsFrom <folder>, or pick a stock model such as yolo11n.pt in Admin > Model.'
} elseif (-not (Test-Path (Join-Path $modelsDir "$defaultModel.json"))) {
  Warn "sidecar\models\$defaultModel.json is missing: without it 'auto' guesses the frame geometry wrong for this weight."
}

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
$shellPolicy = if ($OnWindows) { Get-UserShellPolicy } else { 'Unrestricted' }
if (@('Restricted', 'AllSigned') -contains $shellPolicy) {
  Write-Host ''
  Warn "PowerShell's execution policy is '$shellPolicy', which blocks npm typed into PowerShell."
  Warn 'start.cmd is not affected. To fix PowerShell itself, run this once:'
  Write-Host '    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned'
}

# A desktop shortcut, so launching on a new device is one double-click without finding the repo
# folder. Re-running setup rewrites it, which also repoints it if the folder moved.
$shortcutMade = $false
if ($OnWindows -and -not $NoShortcut -and -not $env:CI) {
  try {
    $desktopDir = [Environment]::GetFolderPath('Desktop')
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path $desktopDir 'SCANnCART.lnk'))
    $link.TargetPath = Join-Path $RepoRoot 'start.cmd'
    $link.WorkingDirectory = $RepoRoot
    $link.Description = 'Launch SCANnCART (desktop app + detection sidecar)'
    $link.Save()
    $shortcutMade = $true
  } catch {
    Warn "Could not create the desktop shortcut ($($_.Exception.Message)); start.cmd still launches the app."
  }
}

Write-Host ''
if ($shortcutMade) {
  Say 'Done. To launch the app, double-click "SCANnCART" on the desktop (or start.cmd in the repo folder).'
} else {
  Say 'Done. To launch the app, double-click start.cmd in the repo folder.'
}
# Tools installed by this run are on the PATH of windows opened *after* it, not of the one that
# started it - `make` typed into that same PowerShell window says "is not recognized".
if ($missing.Count -gt 0) {
  Warn 'Tools were installed just now: close this terminal and open a new one before typing make, npm or uv.'
}
Write-Host 'From a terminal instead:'
Write-Host '    cd desktop; npm run dev     # any shell'
Write-Host '    make dev                    # any shell, once Git is installed (make runs its recipes in Git''s sh)'
