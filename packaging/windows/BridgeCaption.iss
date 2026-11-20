[Setup]
AppId={{E1D66720-F420-45C7-965F-FF58F1918823}
AppName=BridgeCaption
AppVersion={#BridgeVersion}
AppPublisher=BridgeCaption
DefaultDirName={localappdata}\Programs\BridgeCaption
DefaultGroupName=BridgeCaption
OutputDir={#OutputDirectory}
OutputBaseFilename=BridgeCaption-windows-x86_64-setup
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=no
RestartApplications=no
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
WizardStyle=modern

[Files]
Source: "{#DistDirectory}\BridgeCaption\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
Root: HKCU; Subkey: "Software\BridgeCaption\Desktop"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey

[Icons]
Name: "{autoprograms}\BridgeCaption"; Filename: "{app}\BridgeCaption.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\BridgeCaption.exe"; Description: "Launch BridgeCaption"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\BridgeCaption.exe"; WorkingDir: "{app}"; Flags: nowait skipifnotsilent; Check: RelaunchRequested

[Code]
function OpenProcess(Access: Cardinal; Inherit: BOOL; ProcessId: Cardinal): THandle;
  external 'OpenProcess@kernel32.dll stdcall setuponly';
function WaitForSingleObject(Handle: THandle; Timeout: Cardinal): Cardinal;
  external 'WaitForSingleObject@kernel32.dll stdcall setuponly';
function CloseHandle(Handle: THandle): BOOL;
  external 'CloseHandle@kernel32.dll stdcall setuponly';

function InitializeSetup(): Boolean;
var I, ProcessId: Integer;
    Argument: String;
    AppHandle: THandle;
begin
  Result := True;
  for I := 1 to ParamCount do
  begin
    Argument := ParamStr(I);
    if CompareText(Copy(Argument, 1, 9), '/WAITPID=') = 0 then
    begin
      ProcessId := StrToIntDef(Copy(Argument, 10, Length(Argument)), 0);
      if ProcessId <= 0 then
      begin
        Log('Updater rejected an invalid application process ID.');
        Result := False;
        Exit;
      end;
      AppHandle := OpenProcess($00100000, False, ProcessId);  { SYNCHRONIZE }
      if AppHandle = 0 then
      begin
        Result := DLLGetLastError = 87;  { ERROR_INVALID_PARAMETER: process is gone }
        if not Result then
          Log('Updater could not open the previous application process.');
        Exit;
      end;
      try
        Log('Updater is waiting for the previous application process.');
        Result := WaitForSingleObject(AppHandle, 30000) = 0;  { WAIT_OBJECT_0 }
        if not Result then
          Log('Updater timed out waiting for the previous application to exit.');
      finally
        CloseHandle(AppHandle);
      end;
      Exit;
    end;
  end;
end;

function RelaunchRequested: Boolean;
var I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), '/RELAUNCH') = 0 then
      Result := True;
end;
