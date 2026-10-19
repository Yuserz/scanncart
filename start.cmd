@echo off
rem SCANnCART: launch the desktop app (it starts the Python sidecar itself).
rem Run setup.cmd once first. cmd resolves npm.cmd, so PowerShell's execution policy does not apply.
if not exist "%~dp0desktop\node_modules" (
  echo The desktop dependencies are not installed - run setup.cmd first.
  if not defined CI pause
  exit /b 1
)
if not exist "%~dp0desktop\node_modules\electron\path.txt" (
  echo Electron did not finish installing - run setup.cmd again, it repairs it.
  if not defined CI pause
  exit /b 1
)
if not exist "%~dp0sidecar\.venv\Scripts\python.exe" (
  echo The sidecar's Python environment is missing - run setup.cmd first.
  if not defined CI pause
  exit /b 1
)
cd /d "%~dp0desktop"
call npm run dev
if errorlevel 1 if not defined CI pause
