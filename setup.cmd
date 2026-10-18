@echo off
rem SCANnCART one-command setup for Windows: double-click this, or run it from cmd/PowerShell.
rem It runs scripts\setup.ps1, which installs Node.js, uv/Python 3.12, the VC++ runtime, Git and
rem GNU Make (winget, or the tools' own installers), then the desktop and sidecar dependencies.
rem Pass -Check, -ToolsOnly, -Test or -ModelsFrom <folder> through to it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set SETUP_EXIT=%ERRORLEVEL%
if not "%SETUP_EXIT%"=="0" (
  echo.
  echo Setup failed - see the messages above.
)
rem Keep a double-clicked window open to read; CI has no one to press a key.
if not defined CI pause
exit /b %SETUP_EXIT%
