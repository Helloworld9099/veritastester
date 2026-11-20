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
function RelaunchRequested: Boolean;
var I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), '/RELAUNCH') = 0 then
      Result := True;
end;
