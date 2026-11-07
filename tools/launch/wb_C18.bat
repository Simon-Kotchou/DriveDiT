@echo off
rem ==========================================================================================
rem DriveDiT cold-run Workbench launcher: Reforger 1.8.0.13 from the verified working copy.
rem Ticket DD-14a (MIGRATION_1_9 7.1, decision 16; CRIT F-11). Install:
rem E:\DriveDiT\tools\launch\wb_C18.bat. It runs 1.8 beside the 1.9 stable installs, for 1.8
rem baselines and as the preferred rollback. It never reads ACTIVE and never touches the Steam
rem installs on C: or D:.
rem
rem 1. Refuses if Workbench is already running (any build: the two Workbenches share NET API
rem    port 5775).
rem 2. Runs the build-pin preflight in WSL against the EXPLICIT cold pin
rem    E:\DriveDiT\pin\pin_1.8.0.13-cold.json (made with make_pin.py --channel cold), with
rem    --launch-path cold-1.8.0.13. The preflight refuses unless the copy's paks and exes match
rem    the pin (size and mtime; every exe also by sha256) and the pin was made for this launch
rem    path. On success it writes E:\DriveDiT\pin\launches\launch_ID.json (dd-launch/2, channel
rem    stable, pin_channel cold, the 1.8.0.13 build_key) and prints only the launch id.
rem 3. Copies that record to WB_PROFILE\profile\dc_launch\launch_ID.json and checks the copy
rem    byte for byte, as wb_R1.bat does, for DC_LaunchGuard.
rem 4. Sets TEMP and TMP to E:\DriveDiT\tmp\wb_c18 (before WSL and the exe start).
rem 5. Starts the copy's Workbench exe directly with its own profile and its own addon copy
rem    (E:\DriveDiT\addons_c18, deployed with deploy_addon.py --target-pin 1.8.0.13-cold), so a
rem    1.8 Workbench never re-saves the 1.9 addon in E:\DriveDiT\addons.
rem
rem USER STEPS: close Steam or put it in offline mode first (7.1). UNVERIFIED until the first
rem start (decision 16): that the exe runs outside its Steam library at all (the copy holds
rem steam_appid.txt) and that Workbench runs from there.
rem Launch path: WB_DIR, GAME_ADDONS, WB_PROFILE and WB_GPROJ must stay the cold-1.8.0.13 entries
rem of pinlib.COLD_LAUNCH_PATHS, WB_LAUNCH and DEPLOY_ROOTS (tools/pin/tests compares them).
rem Change both together, never one alone.
rem WSL: the distro is named explicitly, because a second distro (Ubuntu-22.04-E) exists.
rem ==========================================================================================
setlocal
set "WB_DIR=E:\DriveDiT\pin\1.8.0.13-run\tools\Workbench"
set "WB_EXE=%WB_DIR%\ArmaReforgerWorkbenchSteamDiag.exe"
set "WB_PROFILE=E:\DriveDiT\wb_profile_c18"
set "WB_GPROJ=E:\DriveDiT\addons_c18\DataCapture\addon.gproj"
set "GAME_ADDONS=E:\DriveDiT\pin\1.8.0.13-run\game\addons"
set "DD_PIN=/mnt/e/DriveDiT/pin/pin_1.8.0.13-cold.json"
set "DD_LAUNCH_PATH=cold-1.8.0.13"
set "DD_LAUNCHES=E:\DriveDiT\pin\launches"
set "DD_TEMP=E:\DriveDiT\tmp\wb_c18"
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

rem TEMP and TMP on E: (R1: a cold run writes nothing to C:, and a normal Windows environment points both at
rem the user's C: profile). The directory is created if missing; the launch stops unless it exists and is on E:.
if not exist "%DD_TEMP%\" mkdir "%DD_TEMP%"
if not exist "%DD_TEMP%\" (
  set "DD_ERR=Cannot create the temp dir %DD_TEMP%. Nothing was started."
  goto :fail
)
if /I not "%DD_TEMP:~0,3%"=="E:\" (
  set "DD_ERR=The temp dir %DD_TEMP% is not on E:. Nothing was started."
  goto :fail
)
set "TEMP=%DD_TEMP%"
set "TMP=%DD_TEMP%"

rem WSL_UTF8=1 and the two findstr filters: see wb_R1.bat. Only a well-formed wb launch id reaches LID.
set "WSL_UTF8=1"
set "LID="
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-wb-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZw]"
for /f %%i in ('wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --exe wb ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. Workbench was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --check"
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
