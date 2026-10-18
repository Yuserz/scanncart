@echo off
rem SCANnCART one-command setup for Windows: double-click this, or run it from cmd/PowerShell.
rem It runs scripts\setup.ps1, which installs Git, Node.js, uv/Python 3.12, GNU Make and the VC++ runtime
rem with winget,
rem then the desktop and sidecar dependencies. Pass -Check, -ToolsOnly or -Test through to it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup failed - see the messages above.
)
pause
