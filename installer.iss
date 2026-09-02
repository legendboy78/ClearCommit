; Inno Setup script for Live Interpreter (per-user install, no admin).
; Build: install Inno Setup 6, then run:
;   ISCC.exe installer.iss      (or double-click build_installer.bat)
; Prerequisite: build the app first -> dist\LiveInterpreter\LiveInterpreter.exe

#define AppName "Live Interpreter"
#define AppVersion "1.0.0"
#define AppPublisher "Voice-end-to-end"
#define AppExe "LiveInterpreter.exe"

[Setup]
AppId={{7F3A1C42-9E5B-4A64-8F2D-LIVEINTERP01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputBaseFilename=LiveInterpreter-Setup-{#AppVersion}
OutputDir=installer
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Keep it clean: no registry garbage, per-user only
UninstallFilesDir={localappdata}\{#AppName}\unins

[Files]
Source: "dist\LiveInterpreter\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\Live Interpreter"; Filename: "{app}\{#AppExe}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; user data (keys, transcripts, logs) in %APPDATA%\LiveInterpreter is NEVER deleted
Type: none; Name: "{app}"
