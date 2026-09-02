@echo off
rem ============================================================
rem  Live Interpreter - full release build
rem  1) PyInstaller freeze   2) Inno Setup installer (if present)
rem  One-time: install Inno Setup 6 from https://jrsoftware.org/isdl.php
rem ============================================================
setlocal
cd /d "%~dp0"

echo [1/2] Freezing app with PyInstaller...
python -m PyInstaller --noconfirm --clean LiveInterpreter.spec
if errorlevel 1 ( echo PyInstaller build FAILED & exit /b 1 )

echo [2/2] Building installer with Inno Setup...
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%ISCC%" (
  "%ISCC%" installer.iss
  if errorlevel 1 ( echo Inno Setup build FAILED & exit /b 1 )
  echo.
  echo DONE. Installer: installer\LiveInterpreter-Setup-1.0.0.exe
) else (
  echo Inno Setup 6 not found - portable build is ready in dist\LiveInterpreter\
  echo Install Inno Setup 6 to also produce the installer.
)
endlocal
