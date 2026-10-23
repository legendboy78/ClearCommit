@echo off
rem ============================================================
rem  ClearCommit - One-click setup + launch
rem  Installs dependencies, checks .env, launches the app.
rem ============================================================
setlocal
cd /d "%~dp0"
title ClearCommit Setup

echo.
echo  ============================================
echo   ClearCommit - Setup and Launch
echo  ============================================
echo.

rem --- 1. Check Python ---
where python >nul 2>nul
if errorlevel 1 (
  echo  [X] Python not found on PATH.
  echo      Install Python 3.12+ from https://www.python.org/downloads/
  echo      Make sure "Add Python to PATH" is checked during install.
  echo.
  pause
  exit /b 1
)
echo  [OK] Python found.

rem --- 2. Install dependencies ---
echo  [*] Installing dependencies (first run only)...
python -c "import groq, customtkinter, numpy, sounddevice, requests" >nul 2>nul
if errorlevel 1 (
  echo  [*] Installing packages from requirements.txt...
  python -m pip install -r requirements.txt --quiet
  if errorlevel 1 (
    echo  [X] Failed to install dependencies.
    echo      Try: python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
  )
  echo  [OK] Dependencies installed.
) else (
  echo  [OK] Dependencies already installed.
)

rem --- 3. Check .env ---
if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo  [!] Created .env from template.
    echo      Edit .env and add your API keys, then relaunch.
    echo      Get free keys at:
    echo        Groq:      https://console.groq.com/keys
    echo        Deepgram:  https://console.deepgram.com
    echo.
    notepad .env
    pause
    exit /b 0
  ) else (
    echo  [!] No .env file found.
    echo      Create one with your API keys:
    echo        GROQ_API_KEY=your_key
    echo        DEEPGRAM_API_KEY=your_key
    echo.
    pause
    exit /b 1
  )
)

rem Check that keys are not still the placeholder
findstr /c:"your_" .env >nul 2>nul
if not errorlevel 1 (
  echo  [!] .env still has placeholder keys.
  echo      Edit .env and replace "your_..._key_here" with real keys.
  echo.
  notepad .env
  echo  [*] Relaunching after .env edit...
)

rem --- 4. Launch ---
echo  [OK] Starting ClearCommit...
echo.
python stream_gui.py
if errorlevel 1 (
  echo.
  echo  [X] ClearCommit exited with an error. Check logs\ for details.
  echo.
  pause
)
endlocal
