@echo off
rem ==========================================================================================
rem DriveDiT cold-run Workbench launcher: Reforger 1.8.0.13 from the verified working copy.
rem Ticket DD-14a (MIGRATION_1_9 7.1, decision 16; CRIT F-11). Install:
rem <DATA_ROOT>\tools\launch\wb_C18.bat, next to dd_local_config.cmd. It runs 1.8 beside the 1.9
rem stable installs, for 1.8 baselines and as the preferred rollback. It never reads ACTIVE and
rem never touches the Steam installs (GAME_INSTALL, TOOLS_INSTALL).
rem
rem Local paths (DD-22): as in wb_R1.bat, every machine path comes from drivedit.local.env through
rem dd_local_config.cmd (DD_DATA_ROOT_WIN, DD_DATA_DRIVE_WIN, DD_DATA_ROOT and DD_WSL_DISTRO here);
rem without it nothing is started. <DATA_ROOT> below is that file's DATA_ROOT.
rem
rem 1. Refuses if Workbench is already running (any build: the two Workbenches share NET API
rem    port 5775).
rem 2. Runs the build-pin preflight in WSL against the EXPLICIT cold pin
rem    <DATA_ROOT>\pin\pin_1.8.0.13-cold.json (made with make_pin.py --channel cold), with
rem    --launch-path cold-1.8.0.13. The preflight refuses unless the copy's paks and exes match
rem    the pin (size and mtime; every exe also by sha256) and the pin was made for this launch
rem    path. On success it writes <DATA_ROOT>\pin\launches\launch_ID.json (dd-launch/2, channel
rem    stable, pin_channel cold, the 1.8.0.13 build_key) and prints only the launch id.
rem 3. Copies that record to WB_PROFILE\profile\dc_launch\launch_ID.json and checks the copy
rem    byte for byte, as wb_R1.bat does, for DC_LaunchGuard.
rem 4. Sets TEMP and TMP to <DATA_ROOT>\tmp\wb_c18 (before WSL and the exe start).
rem 5. Starts the copy's Workbench exe directly with its own profile and its own addon copy
rem    (<DATA_ROOT>\addons_c18, deployed with deploy_addon.py --target-pin 1.8.0.13-cold), so a
rem    1.8 Workbench never re-saves the 1.9 addon in <DATA_ROOT>\addons.
rem
rem USER STEPS: close Steam or put it in offline mode first (7.1). UNVERIFIED until the first
rem start (decision 16): that the exe runs outside its Steam library at all (the copy holds
rem steam_appid.txt) and that Workbench runs from there.
rem Launch path: WB_DIR, GAME_ADDONS, WB_PROFILE and WB_GPROJ must stay the cold-1.8.0.13 entries
rem of pinlib.COLD_LAUNCH_PATHS, WB_LAUNCH and DEPLOY_ROOTS (tools/pin/tests compares them).
rem Both read DATA_ROOT of the same local config, so they cannot drift.
rem WSL: the distro is named explicitly (WSL_DISTRO of the local config), because a machine may have
rem more than one.
rem ==========================================================================================
setlocal
if not exist "%~dp0dd_local_config.cmd" (
  set "DD_ERR=Missing %~dp0dd_local_config.cmd: install it next to this launcher (tools\launch). Nothing was started."
  goto :fail
)
call "%~dp0dd_local_config.cmd"
if errorlevel 1 goto :fail
set "WB_DIR=%DD_DATA_ROOT_WIN%\pin\1.8.0.13-run\tools\Workbench"
set "WB_EXE=%WB_DIR%\ArmaReforgerWorkbenchSteamDiag.exe"
set "WB_PROFILE=%DD_DATA_ROOT_WIN%\wb_profile_c18"
set "WB_GPROJ=%DD_DATA_ROOT_WIN%\addons_c18\DataCapture\addon.gproj"
set "GAME_ADDONS=%DD_DATA_ROOT_WIN%\pin\1.8.0.13-run\game\addons"
set "DD_PIN=%DD_DATA_ROOT%/pin/pin_1.8.0.13-cold.json"
set "DD_LAUNCH_PATH=cold-1.8.0.13"
set "DD_LAUNCHES=%DD_DATA_ROOT_WIN%\pin\launches"
set "DD_TEMP=%DD_DATA_ROOT_WIN%\tmp\wb_c18"
set "DD_ERR="

rem /FO CSV: the default table format truncates image names to 25 characters.
tasklist /FI "IMAGENAME eq ArmaReforgerWorkbenchSteamDiag.exe" /FO CSV /NH 2>nul | find /I "ArmaReforgerWorkbenchSteamDiag.exe" >nul
if not errorlevel 1 (
  set "DD_ERR=Workbench is already running. Save your work, close it, then run this launcher again."
  goto :fail
)
if not exist "%WB_EXE%" (
  set "DD_ERR=Workbench exe not found: %WB_EXE%. Make the working copy first (tools\pin\README.md, cold-run pin)."
  goto :fail
)
if not exist "%GAME_ADDONS%\data\ArmaReforger.gproj" (
  set "DD_ERR=Game addons not found: %GAME_ADDONS%\data\ArmaReforger.gproj"
  goto :fail
)
if not exist "%WB_GPROJ%" (
  set "DD_ERR=Missing %WB_GPROJ%. Deploy the addon with deploy_addon.py --target-pin 1.8.0.13-cold first."
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

rem WSL_UTF8=1 and the two findstr filters: see wb_R1.bat. Only a well-formed wb launch id reaches LID.
set "WSL_UTF8=1"
set "LID="
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-wb-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZw]"
for /f %%i in ('wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B %DD_DATA_ROOT%/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --exe wb ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. Workbench was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B %DD_DATA_ROOT%/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --check"
  goto :fail
)
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
  goto :fail
)

set "WB_DC_LAUNCH=%WB_PROFILE%\profile\dc_launch"
if not exist "%WB_DC_LAUNCH%\" mkdir "%WB_DC_LAUNCH%"
if not exist "%WB_DC_LAUNCH%\" (
  set "DD_ERR=Cannot create %WB_DC_LAUNCH%. Workbench was not started."
  goto :fail
)
if exist "%WB_DC_LAUNCH%\launch_%LID%.json" (
  set "DD_ERR=A copy of launch record %LID% already exists in %WB_DC_LAUNCH%. Workbench was not started."
  goto :fail
)
copy "%DD_LAUNCHES%\launch_%LID%.json" "%WB_DC_LAUNCH%\launch_%LID%.json" >nul
fc /b "%DD_LAUNCHES%\launch_%LID%.json" "%WB_DC_LAUNCH%\launch_%LID%.json" >nul 2>nul
if errorlevel 1 (
  set "DD_ERR=Could not copy launch record %LID% to %WB_DC_LAUNCH% - the copy is missing or differs. Workbench was not started."
  goto :fail
)

echo Pin OK (cold 1.8.0.13). Launch id %LID%
start "" /D "%WB_DIR%" "%WB_EXE%" -profile "%WB_PROFILE%" -gproj "%WB_GPROJ%" -addonsDir "%GAME_ADDONS%" -forceUpdate -maxFPS 60 -dcLaunchId %LID%
endlocal
exit /b 0

:fail
echo %DD_ERR%
pause
endlocal
exit /b 1
