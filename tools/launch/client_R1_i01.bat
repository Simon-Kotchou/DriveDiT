@echo off
rem ==========================================================================================
rem DriveDiT pinned game-client launcher, rig R1, instance i01.
rem Ticket DD-0 (capture program r2 TKT-0, Appendix B; decision D1).
rem Install: E:\DriveDiT\tools\launch\client_R1_i01.bat. Copy it to client_R1_iNN.bat for more
rem instances, changing DD_INSTANCE, -posX/-posY and -GPUAdapter (S2).
rem
rem 1. Refuses until the per-build settings below are filled in and engine_R1.conf exists.
rem 2. Runs the build-pin preflight in WSL with --pin-active (see wb_R1.bat): it checks the pin
rem    that E:\DriveDiT\pin\ACTIVE names. It prints a launch id on success and writes
rem    E:\DriveDiT\pin\launches\launch_ID.json, which records the pin id and how it was
rem    resolved; it prints nothing on failure. The launch logic names no game version.
rem 3. Starts ArmaReforgerSteam.exe directly. The working directory is the game dir, as in a
rem    Steam launch (its steam_appid.txt holds 1874880). The launch id goes to the game as
rem    -dcLaunchId, which joins the session to the preflight record (capture program 3.9).
rem No pause on failure: a supervisor may run this unattended (exit code 1).
rem Build: 1.8.0.13 is the INTERIM pin. The plan is to move to 1.9 and its new vehicle physics
rem once 1.9 is stable and compatibility is checked. See program\1_9\MIGRATION_1_9.md in
rem E:\drivedit_realign once written, and tools\pin\README.md. Switch pins only with
rem E:\DriveDiT\tools\pin\pin_activate.py. Never edit ACTIVE by hand.
rem
rem Flag provenance. The flags are those of Appendix B. They come from the Startup Parameters
rem wiki page, revision 2025-02-14 (trial-ladder.md 8.2, source S1), and are UNVERIFIED on
rem 1.8.0.13 until probe S1. Re-verify every flag on the new build before capturing on 1.9:
rem   -cfg            "forces to load specific user engine settings config". The file is built
rem                   by probe E-20; whether the engine applies it is checked at S1.
rem   -screenWidth/-screenHeight  set the WINDOW size, not the render resolution. The size is
rem                   1408x720, the r2 source class (a 1280x720 core plus 64 px bands). A PNG of
rem                   exactly 1408x720 is UNVERIFIED (P-03).
rem   -forceUpdate    renders while unfocused. Fresh frames when the window is covered are
rem                   UNVERIFIED (P-13).
rem   -GPUAdapter 0   which physical GPU index 0 maps to is UNVERIFIED (S2).
rem   -dcLaunchId     is custom. The capture scripts must read it with System.GetCLIParam
rem                   (TKT-4); until then the game does not use it.
rem Addon/world arguments are from trial-ladder.md 8.2:
rem   -addonsDir      E:\DriveDiT\addons holds the UNPACKED DataCapture addon (DD-1 deploy). It is
rem                   UNVERIFIED that the client loads an unpacked addon (P-02). If it does not,
rem                   pack to E:\DriveDiT\build\DataCapture_SHA8 and point DC_ADDONS_DIR there.
rem   -addons         is the GUID of the DataCapture addon, 88BEF4187E0D2DFC.
rem   -world          is UNSET until TKT-2-min creates DC_Arland.ent. The format is
rem                   {GUID}worlds/DataCapture/DC_Arland.ent. It is UNVERIFIED that -world starts
rem                   the game mode on 1.8 (P-12).
rem   -noBackend      is intentionally absent. Add it only for capture-only runs (T5), never for
rem                   closed-loop runs (T6), per trial-ladder.md 8.2.
rem UNVERIFIED (capture program 4.2 item 2, P-06): that a direct exe launch never makes Steam
rem update or relaunch the game, and that Steam allows concurrent instances. Steam must be running.
rem ==========================================================================================
setlocal
set "DD_INSTANCE=i01"
set "GAME_DIR=D:\SteamLibrary\steamapps\common\Arma Reforger"
set "GAME_EXE=%GAME_DIR%\ArmaReforgerSteam.exe"
set "DC_PROFILE=E:\DriveDiT\client_profiles\%DD_INSTANCE%"
set "DC_LOGS=E:\DriveDiT\logs\%DD_INSTANCE%"
set "DC_CFG=E:\DriveDiT\tools\cfg\engine_R1.conf"
set "DD_LAUNCHES=E:\DriveDiT\pin\launches"
rem ---- per-build settings (trial-ladder.md 8.2) ----
set "DC_ADDONS_DIR=E:\DriveDiT\addons"
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
  set "DD_ERR=Game exe not found: %GAME_EXE%"
  goto :fail
)
if not exist "%DC_ADDONS_DIR%\" (
  set "DD_ERR=Addons dir not found: %DC_ADDONS_DIR%"
  goto :fail
)

rem WSL_UTF8=1: wsl.exe writes its own messages (for example a missing distro) to STDOUT, in UTF-16 by
rem default (measured 2026-10-20). UTF-8 keeps such a line from garbling the line after it.
set "WSL_UTF8=1"
set "LID="
rem Only a stdout line that starts with a launch id (YYYYMMDDTHHMMSSZ-client-xxxx) and holds nothing but
rem id characters reaches LID. Every other line (a wsl.exe message, a stray print) is dropped, so it
rem can neither block a good launch nor put cmd metacharacters (ampersand, pipe, angle brackets,
rem quotes, percent) into LID. findstr /X and $ never match LF-only lines (measured), hence two
rem filters: the id pattern anchored at the start, then /V on any character outside the id set.
rem The interpreter is named (/usr/bin/python3) because -e starts no shell and its PATH holds Windows dirs.
set "DD_ID_RE=^[0123456789][0123456789]*T[0123456789][0123456789]*Z-client-[0123456789abcdef][0123456789abcdef]*"
set "DD_ID_BAD=[^-0123456789abcdefTZilnt]"
rem preflight prints a launch id on success and writes launch_ID.json; it prints nothing on failure
for /f %%i in ('wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --exe client --instance %DD_INSTANCE% ^| findstr /R /C:"%DD_ID_RE%" ^| findstr /R /V /C:"%DD_ID_BAD%"') do set "LID=%%i"
if not defined LID (
  set "DD_ERR=PIN CHECK FAILED. The client was not started. The preflight messages above say why. If there are none, wsl.exe itself failed: run wsl.exe -d Ubuntu-22.04 -e /usr/bin/python3 -I -B /mnt/e/DriveDiT/tools/pin/preflight.py --pin-active --check"
  goto :fail
)
rem The record must exist: a launch id alone is not proof that this preflight run passed.
if not exist "%DD_LAUNCHES%\launch_%LID%.json" (
  set "DD_ERR=PIN CHECK FAILED. No launch record for id %LID% in %DD_LAUNCHES%."
  goto :fail
)

if not exist "%DC_LOGS%\" mkdir "%DC_LOGS%"
echo Pin OK. Launch id %LID%
start "" /D "%GAME_DIR%" "%GAME_EXE%" -profile "%DC_PROFILE%" -logsDir "%DC_LOGS%" -cfg "%DC_CFG%" -window -screenWidth 1408 -screenHeight 720 -posX 0 -posY 0 -forceUpdate -noFocus -maxFPS 60 -GPUAdapter 0 -freezeCheck 120 -freezeCheckMode kill -noThrow -dcLaunchId %LID% -addonsDir "%DC_ADDONS_DIR%" -addons %DC_ADDON_GUID% -world "%DC_WORLD%"
endlocal
exit /b 0

:fail
echo %DD_ERR%
endlocal
exit /b 1
