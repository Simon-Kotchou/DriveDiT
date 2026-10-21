@echo off
rem ==========================================================================================
rem DriveDiT pinned Workbench launcher, rig R1.
rem Ticket DD-0 (capture program r2 TKT-0, Appendix B; decision D1).
rem Install: E:\DriveDiT\tools\launch\wb_R1.bat. Never launch Workbench from the Steam library
rem during the campaign.
rem
rem 1. Refuses if Workbench is already running (it never starts a second instance).
rem 2. Runs the build-pin preflight in WSL. The preflight refuses unless the Steam appmanifests
rem    and the installed files of BOTH Arma Reforger and Arma Reforger Tools match
rem    E:\DriveDiT\pin\pin_1.8.0.13.json, and AutoUpdateBehavior is "1" ("Only update this game
rem    when I launch it"). On success it writes E:\DriveDiT\pin\launches\launch_ID.json and
rem    prints only the launch id.
rem 3. Starts the Workbench exe directly. The working directory is the Workbench dir, as in a
rem    Steam launch (its steam_appid.txt holds 1874910).
rem
rem UNVERIFIED (capture program 4.2 item 2): that a direct exe launch never makes Steam update
rem or relaunch the app. Steam must be running.
rem UNVERIFIED (FIX_PLAN step 4, gate P-01): that Workbench honours -profile. After the first
rem start, check console.log for "Adding relative directory 'E:\DriveDiT\wb_profile\profile'".
rem Args (Appendix B and FIX_PLAN step 4a): -profile, -gproj, -forceUpdate, -maxFPS 60.
rem WSL: the distro is named explicitly, because a second distro (Ubuntu-22.04-E) exists.
rem ==========================================================================================
setlocal
set "WB_DIR=C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench"
set "WB_EXE=%WB_DIR%\ArmaReforgerWorkbenchSteamDiag.exe"
set "WB_PROFILE=E:\DriveDiT\wb_profile"
set "WB_GPROJ=E:\DriveDiT\addons\DataCapture\addon.gproj"
set "DD_LAUNCHES=E:\DriveDiT\pin\launches"
set "DD_ERR="

rem /FO CSV: the default table format truncates image names to 25 characters.
tasklist /FI "IMAGENAME eq ArmaReforgerWorkbenchSteamDiag.exe" /FO CSV /NH 2>nul | find /I "ArmaReforgerWorkbenchSteamDiag.exe" >nul
if not errorlevel 1 (
  set "DD_ERR=Workbench is already running. Save your work, close it, then run this launcher again."
  goto :fail
)
if not exist "%WB_EXE%" (
  set "DD_ERR=Workbench exe not found: %WB_EXE%"
  goto :fail
)
if not exist "%WB_GPROJ%" (
  set "DD_ERR=Missing %WB_GPROJ%. Deploy the DataCapture addon first, ticket DD-1."
  goto :fail
)

set "LID="
for /f %%i in ('wsl.exe -d Ubuntu-22.04 -e python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin /mnt/e/DriveDiT/pin/pin_1.8.0.13.json --exe wb') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. Workbench was not started. The preflight messages above say why."
  goto :fail
)
rem The record must exist: this also rejects a stray stdout line (for example a wsl.exe warning).
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
  goto :fail
)

echo Pin OK. Launch id %LID%
start "" /D "%WB_DIR%" "%WB_EXE%" -profile "%WB_PROFILE%" -gproj "%WB_GPROJ%" -forceUpdate -maxFPS 60
endlocal
exit /b 0

:fail
echo %DD_ERR%
pause
endlocal
exit /b 1
