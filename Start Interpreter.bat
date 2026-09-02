@echo off
rem Live Interpreter launcher - double-click to start.
rem Devices/settings are remembered from the last session (settings.json).
cd /d "%~dp0"
title Live Interpreter
python stream_gui.py %*
if errorlevel 1 (
  echo.
  echo Interpreter exited with an error. See logs\ for the session log.
  pause
)
