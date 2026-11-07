@echo off
rem ==========================================================================================
rem DriveDiT cold-run game-client launcher: Reforger 1.8.0.13 from the verified working copy,
rem instance c18_i01. Ticket DD-14a (MIGRATION_1_9 7.1, decision 16; CRIT F-11). Install:
rem E:\DriveDiT\tools\launch\client_C18_i01.bat. It never reads ACTIVE.
rem
rem As client_R1_i01.bat, except:
rem   - the preflight checks the EXPLICIT cold pin E:\DriveDiT\pin\pin_1.8.0.13-cold.json with
rem     --launch-path cold-1.8.0.13 (the copy, checked against its pin and SHA256SUMS hashes);
rem   - GAME_DIR is the working copy E:\DriveDiT\pin\1.8.0.13-run\game; profile, logs and the
rem     addon copy (E:\DriveDiT\addons_c18) are its own;
rem   - -noBackend is ALWAYS passed: after 1.9 is live a 1.8 client may be refused or degraded by
rem     the production backend (7.1 [inference, UNVERIFIED]); the cold pin records it in
rem     client_args_required.
rem No pause on failure: a supervisor may run this unattended (exit code 1).
rem USER STEPS: close Steam or put it in offline mode first (7.1). UNVERIFIED until the first
rem start (decision 16): that the exe runs outside its Steam library at all.
rem Launch path: GAME_DIR and DC_ADDONS_DIR must stay the cold-1.8.0.13 entries of
rem pinlib.COLD_LAUNCH_PATHS and DEPLOY_ROOTS (tools/pin/tests compares them).
rem The flags are those of client_R1_i01.bat (see its provenance notes), plus -noBackend.
rem ==========================================================================================
setlocal
set "DD_INSTANCE=c18_i01"
set "GAME_DIR=E:\DriveDiT\pin\1.8.0.13-run\game"
set "GAME_EXE=%GAME_DIR%\ArmaReforgerSteam.exe"
set "DC_PROFILE=E:\DriveDiT\client_profiles\%DD_INSTANCE%"
set "DC_LOGS=E:\DriveDiT\logs\%DD_INSTANCE%"
set "DC_CFG=E:\DriveDiT\tools\cfg\engine_R1.conf"
set "DD_PIN=/mnt/e/DriveDiT/pin/pin_1.8.0.13-cold.json"
set "DD_LAUNCH_PATH=cold-1.8.0.13"
set "DD_LAUNCHES=E:\DriveDiT\pin\launches"
rem ---- per-build settings (trial-ladder.md 8.2) ----
set "DC_ADDONS_DIR=E:\DriveDiT\addons_c18"
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

rem WSL_UTF8=1 and the two findstr filters: see client_R1_i01.bat. Only a well-formed client launch id reaches LID.
set "WSL_UTF8=1"
set "LID="
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-client-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZilnt]"
for /f %%i in ('wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --exe client --instance %DD_INSTANCE% ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. The client was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin %DD_PIN% --launch-path %DD_LAUNCH_PATH% --check"
  goto :fail
)
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
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
