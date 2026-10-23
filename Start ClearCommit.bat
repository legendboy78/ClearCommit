@echo off
rem ============================================================
rem  ClearCommit (Live Interpreter) - double-click to start.
rem  Two-mode communication assistant on top of the live
rem  translator: Social / Professional modes, AI coaching,
rem  meeting intelligence and the agreement ledger.
rem  Devices/settings are remembered from the last session.
rem ============================================================
setlocal
cd /d "%~dp0"
title ClearCommit Interpreter

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python was not found on PATH.
  echo  Install Python 3.12+ from https://www.python.org/downloads/
  echo  and re-run this launcher.
  echo.
  pause
  exit /b 1
)

if not exist "stream_gui.py" (
  echo.
  echo  stream_gui.py not found next to this launcher.
  echo  Run this file from the application folder.
  echo.
  pause
  exit /b 1
)

python stream_gui.py %*
if errorlevel 1 (
  echo.
  echo  ClearCommit exited with an error. See logs\ for the session log.
  echo.
  pause
)
endlocal
