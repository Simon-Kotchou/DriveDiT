@echo off
rem ==========================================================================================
rem DriveDiT pinned Workbench launcher, rig R1.
rem Ticket DD-0 (capture program r2 TKT-0, Appendix B; decision D1).
rem Install: E:\DriveDiT\tools\launch\wb_R1.bat. Never launch Workbench from the Steam library
rem during the campaign.
rem
rem 1. Refuses if Workbench is already running (it never starts a second instance).
rem 2. Runs the build-pin preflight in WSL with --pin-active: E:\DriveDiT\pin\ACTIVE names the
rem    active pin id, and the pin is E:\DriveDiT\pin\pin_ID.json. The preflight refuses unless
rem    ACTIVE is valid and the Steam appmanifests and the installed files of BOTH Arma Reforger and
rem    Arma Reforger Tools match that pin, and AutoUpdateBehavior is "1" ("Only update this game
rem    when I launch it"). On success it writes E:\DriveDiT\pin\launches\launch_ID.json and
rem    prints only the launch id. The launch logic names no game version.
rem 3. Starts the Workbench exe directly. The working directory is the Workbench dir, as in a
rem    Steam launch (its steam_appid.txt holds 1874910).
rem
rem UNVERIFIED (capture program 4.2 item 2): that a direct exe launch never makes Steam update
rem or relaunch the app. Steam must be running.
rem UNVERIFIED (FIX_PLAN step 4, gate P-01): that Workbench honours -profile. After the first
rem start, check console.log for "Adding relative directory 'E:\DriveDiT\wb_profile\profile'".
rem Args (Appendix B and FIX_PLAN step 4a): -profile, -gproj, -addonsDir, -forceUpdate, -maxFPS 60.
rem -addonsDir: a direct exe launch does not get the game install's addons dir the way a Steam launch
rem does, so ArmaReforger.gproj (58D0FB3206B6F859) is "not found" and Workbench exits with "Cannot
rem initialize game project settings" (measured 2026-10-21 18:50, first wb_R1.bat launch). GAME_ADDONS
rem is the pinned game install (pinlib.LAUNCH_PATHS "game") plus \addons; it holds core\ and data\.
rem Build: 1.8.0.13 is the INTERIM pin. The plan is to move to 1.9 and its new vehicle physics
rem once 1.9 is stable and compatibility is checked. See program\1_9\MIGRATION_1_9.md in
rem E:\drivedit_realign once written, and tools\pin\README.md. Switch pins only with
rem E:\DriveDiT\tools\pin\pin_activate.py, which refuses a pin that does not match the installs.
rem Never edit ACTIVE by hand.
rem Launch path: WB_DIR must stay the Tools install named in pinlib.LAUNCH_PATHS (plus \Workbench).
rem The preflight check tools.launch_path refuses any pin not made from that install, so the exe
rem started here is always the one the pin checked. Change both together, never one alone.
rem WSL: the distro is named explicitly, because a second distro (Ubuntu-22.04-E) exists.
rem ==========================================================================================
setlocal
set "WB_DIR=C:\Program Files (x86)\Steam\steamapps\common\Arma Reforger Tools\Workbench"
set "WB_EXE=%WB_DIR%\ArmaReforgerWorkbenchSteamDiag.exe"
set "WB_PROFILE=E:\DriveDiT\wb_profile"
set "WB_GPROJ=E:\DriveDiT\addons\DataCapture\addon.gproj"
set "GAME_ADDONS=D:\SteamLibrary\steamapps\common\Arma Reforger\addons"
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
if not exist "%GAME_ADDONS%\data\ArmaReforger.gproj" (
  set "DD_ERR=Game addons not found: %GAME_ADDONS%\data\ArmaReforger.gproj"
  goto :fail
)
if not exist "%WB_GPROJ%" (
  set "DD_ERR=Missing %WB_GPROJ%. Deploy the DataCapture addon first, ticket DD-1."
  goto :fail
)

rem WSL_UTF8=1: wsl.exe writes its own messages (for example a missing distro) to STDOUT, in UTF-16 by
rem default (measured 2026-10-20). UTF-8 keeps such a line from garbling the line after it.
set "WSL_UTF8=1"
set "LID="
rem Only a stdout line that starts with a launch id (YYYYMMDDTHHMMSSZ-wb-xxxx) and holds nothing but
rem id characters reaches LID. Every other line (a wsl.exe message, a stray print) is dropped, so it
rem can neither block a good launch nor put cmd metacharacters (ampersand, pipe, angle brackets,
rem quotes, percent) into LID. findstr /X and $ never match LF-only lines (measured), hence two
rem filters: the id pattern anchored at the start, then /V on any character outside the id set.
rem The interpreter is named (/usr/bin/python3) because -e starts no shell and its PATH holds Windows dirs.
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-wb-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZw]"
for /f %%i in ('wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --exe wb ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. Workbench was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --check"
  goto :fail
)
rem The record must exist: a launch id alone is not proof that this preflight run passed.
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
  goto :fail
)

echo Pin OK. Launch id %LID%
start "" /D "%WB_DIR%" "%WB_EXE%" -profile "%WB_PROFILE%" -gproj "%WB_GPROJ%" -addonsDir "%GAME_ADDONS%" -forceUpdate -maxFPS 60
endlocal
exit /b 0

:fail
echo %DD_ERR%
pause
endlocal
exit /b 1
