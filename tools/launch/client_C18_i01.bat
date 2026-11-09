@echo off
rem ==========================================================================================
rem DriveDiT cold-run game-client launcher: Reforger 1.8.0.13 from the verified working copy,
rem instance c18_i01. Ticket DD-14a (MIGRATION_1_9 7.1, decision 16; CRIT F-11). Install:
rem <DATA_ROOT>\tools\launch\client_C18_i01.bat, next to dd_local_config.cmd. It never reads ACTIVE.
rem
rem Local paths (DD-22): as in wb_R1.bat, every machine path comes from drivedit.local.env through
rem dd_local_config.cmd (DD_DATA_ROOT_WIN, DD_DATA_DRIVE_WIN, DD_DATA_ROOT and DD_WSL_DISTRO here);
rem without it nothing is started. <DATA_ROOT> below is that file's DATA_ROOT.
rem
rem As client_R1_i01.bat, except:
rem   - the preflight checks the EXPLICIT cold pin <DATA_ROOT>\pin\pin_1.8.0.13-cold.json with
rem     --launch-path cold-1.8.0.13 (the copy, checked against its pin and SHA256SUMS hashes);
rem   - GAME_DIR is the working copy <DATA_ROOT>\pin\1.8.0.13-run\game; profile, logs and the
rem     addon copy (<DATA_ROOT>\addons_c18) are its own;
rem   - TEMP and TMP are <DATA_ROOT>\tmp\c18_i01, and the launch record is copied into the profile's
rem     dc_launch\ folder and byte-checked before the start, as wb_R1.bat does;
rem   - -noBackend is ALWAYS passed: after 1.9 is live a 1.8 client may be refused or degraded by
rem     the production backend (7.1 [inference, UNVERIFIED]); the cold pin records it in
rem     client_args_required.
rem No pause on failure: a supervisor may run this unattended (exit code 1).
rem USER STEPS: close Steam or put it in offline mode first (7.1). UNVERIFIED until the first
rem start (decision 16): that the exe runs outside its Steam library at all.
rem Launch path: GAME_DIR and DC_ADDONS_DIR must stay the cold-1.8.0.13 entries of
rem pinlib.COLD_LAUNCH_PATHS and DEPLOY_ROOTS (tools/pin/tests compares them); both read DATA_ROOT
rem of the same local config.
rem The flags are those of client_R1_i01.bat (see its provenance notes), plus -noBackend.
rem ==========================================================================================
setlocal
if not exist "%~dp0dd_local_config.cmd" (
  set "DD_ERR=Missing %~dp0dd_local_config.cmd: install it next to this launcher (tools\launch). Nothing was started."
  goto :fail
)
call "%~dp0dd_local_config.cmd"
if errorlevel 1 goto :fail
set "DD_INSTANCE=c18_i01"
set "GAME_DIR=%DD_DATA_ROOT_WIN%\pin\1.8.0.13-run\game"
set "GAME_EXE=%GAME_DIR%\ArmaReforgerSteam.exe"
set "DC_PROFILE=%DD_DATA_ROOT_WIN%\client_profiles\%DD_INSTANCE%"
set "DC_LOGS=%DD_DATA_ROOT_WIN%\logs\%DD_INSTANCE%"
set "DC_CFG=%DD_DATA_ROOT_WIN%\tools\cfg\engine_R1.conf"
set "DD_PIN=%DD_DATA_ROOT%/pin/pin_1.8.0.13-cold.json"
set "DD_LAUNCH_PATH=cold-1.8.0.13"
set "DD_LAUNCHES=%DD_DATA_ROOT_WIN%\pin\launches"
set "DD_TEMP=%DD_DATA_ROOT_WIN%\tmp\%DD_INSTANCE%"
rem ---- per-build settings (trial-ladder.md 8.2) ----
set "DC_ADDONS_DIR=%DD_DATA_ROOT_WIN%\addons_c18"
set "DC_ADDON_GUID=88BEF4187E0D2DFC"
set "DC_WORLD=UNSET"
rem ---------------------------------------------------
set "DD_ERR="

if /I "%DC_WORLD%"=="UNSET" (
  set "DD_ERR=DC_WORLD is UNSET in this launcher. Fill it in once DC_Arland.ent exists, ticket TKT-2-min."
  goto :fail
)
if not exist "%DC_CFG%" (
  set "DD_ERR=Missing %DC_CFG%. Build it in probe E-20 first."
  goto :fail
)
if not exist "%GAME_EXE%" (
  set "DD_ERR=Game exe not found: %GAME_EXE%. Make the working copy first (tools\pin\README.md, cold-run pin)."
  goto :fail
)
if not exist "%DC_ADDONS_DIR%\" (
  set "DD_ERR=Addons dir not found: %DC_ADDONS_DIR%"
  goto :fail
)

rem TEMP and TMP on the data drive (R1: a cold run writes nothing to the system drive, and a normal Windows
rem environment points both at the user's profile there). The directory is created if missing; the launch stops
rem unless it exists and is on the data drive (DD_DATA_DRIVE_WIN, the drive of DATA_ROOT).
if not exist "%DD_TEMP%\" mkdir "%DD_TEMP%"
if not exist "%DD_TEMP%\" (
  set "DD_ERR=Cannot create the temp dir %DD_TEMP%. Nothing was started."
  goto :fail
)
if /I not "%DD_TEMP:~0,3%"=="%DD_DATA_DRIVE_WIN%\" (
  set "DD_ERR=The temp dir %DD_TEMP% is not on the data drive %DD_DATA_DRIVE_WIN%. Nothing was started."
  goto :fail
)
set "TEMP=%DD_TEMP%"
set "TMP=%DD_TEMP%"

rem WSL_UTF8=1 and the two findstr filters: see client_R1_i01.bat. Only a well-formed client launch id reaches LID.
set "WSL_UTF8=1"
set "LID="
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-client-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZilnt]"
for /f %%i in ('wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B %DD_DATA_ROOT%/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --exe client --instance %DD_INSTANCE% ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. The client was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B %DD_DATA_ROOT%/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --check"
  goto :fail
)
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
  goto :fail
)

rem Copy the record into the client profile for DC_LaunchGuard, as wb_R1.bat does for Workbench: $profile is the
rem -profile folder plus \profile (measured for Workbench; UNVERIFIED for the client until its first start, when the
rem run header's launch.profile_abs shows it). The id is new for every launch, so an existing copy is refused, and
rem fc /b proves the copy is identical.
set "DC_DC_LAUNCH=%DC_PROFILE%\profile\dc_launch"
if not exist "%DC_DC_LAUNCH%\" mkdir "%DC_DC_LAUNCH%"
if not exist "%DC_DC_LAUNCH%\" (
  set "DD_ERR=Cannot create %DC_DC_LAUNCH%. The client was not started."
  goto :fail
)
if exist "%DC_DC_LAUNCH%\launch_%LID%.json" (
  set "DD_ERR=A copy of launch record %LID% already exists in %DC_DC_LAUNCH%. The client was not started."
  goto :fail
)
copy "%DD_LAUNCHES%\launch_%LID%.json" "%DC_DC_LAUNCH%\launch_%LID%.json" >nul
fc /b "%DD_LAUNCHES%\launch_%LID%.json" "%DC_DC_LAUNCH%\launch_%LID%.json" >nul 2>nul
if errorlevel 1 (
  set "DD_ERR=Could not copy launch record %LID% to %DC_DC_LAUNCH% - the copy is missing or differs. The client was not started."
  goto :fail
)

if not exist "%DC_LOGS%\" mkdir "%DC_LOGS%"
echo Pin OK (cold 1.8.0.13). Launch id %LID%
start "" /D "%GAME_DIR%" "%GAME_EXE%" -profile "%DC_PROFILE%" -logsDir "%DC_LOGS%" -cfg "%DC_CFG%" -window -screenWidth 1408 -screenHeight 720 -posX 0 -posY 0 -forceUpdate -noFocus -maxFPS 60 -GPUAdapter 0 -freezeCheck 120 -freezeCheckMode kill -noThrow -noBackend -dcLaunchId %LID% -addonsDir "%DC_ADDONS_DIR%" -addons %DC_ADDON_GUID% -world "%DC_WORLD%"
endlocal
exit /b 0

:fail
echo %DD_ERR%
endlocal
exit /b 1
