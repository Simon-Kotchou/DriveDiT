@echo off
rem ==========================================================================================
rem DriveDiT local paths for the Windows launchers (ticket DD-22). Every launcher in this folder
rem CALLs it right after setlocal:   call "%~dp0dd_local_config.cmd"
rem It reads drivedit.local.env, the one untracked file that holds this machine's paths, from
rem the first of these that exists (the lookup order of tools/local_config.py):
rem   1. %DRIVEDIT_LOCAL_CONFIG% (the Windows path of the file);
rem   2. config\drivedit.local.env in the folder above tools\ (two levels above this folder;
rem      for the installed tools that is DATA_ROOT\config\drivedit.local.env);
rem   3. drivedit.local.env in that same folder, only when it is a git checkout (has .git).
rem It sets, in the caller's environment:
rem   DD_LOCAL_CONFIG       the file it read (Windows path)
rem   DD_WSL_DISTRO         WSL_DISTRO, the distro the preflight runs in (wsl.exe -d)
rem   DD_DATA_ROOT          DATA_ROOT, WSL form (the launchers run DD_DATA_ROOT/tools/pin/preflight.py)
rem   DD_DATA_ROOT_WIN      DATA_ROOT, Windows form, and DD_DATA_DRIVE_WIN its drive (for example X:)
rem   DD_GAME_INSTALL_WIN   GAME_INSTALL, Windows form
rem   DD_TOOLS_INSTALL_WIN  TOOLS_INSTALL, Windows form
rem The Windows form is the pure mapping of tools/local_config.py: /mnt/x/rest becomes X:\rest.
rem It also sets DRIVEDIT_LOCAL_CONFIG to the file it read and adds DRIVEDIT_LOCAL_CONFIG/p to
rem WSLENV, so the preflight in WSL reads the very same file.
rem Refuses (exit code 1, DD_ERR says why) when no file is found, or a key it needs is missing or
rem malformed: WSL_DISTRO and DATA_ROOT reach the wsl.exe command line unquoted, so they must
rem match [A-Za-z0-9._-] and /mnt/x/[A-Za-z0-9_./-] exactly, and the installs may use letters,
rem digits, space and _ . , + ( ) ~ / - only (the rules of tools/local_config.py).
rem It reads one file and writes nothing. "dd_local_config.cmd /print" prints the DD_ values
rem instead of returning them (tools/local_config.py check --cmd compares them with Python's).
rem ==========================================================================================
set "DD_ERR="
set "DD_LOCAL_CONFIG="
for %%V in (DATA_ROOT GAME_INSTALL TOOLS_INSTALL WSL_DISTRO K P L U R W ROOT) do set "DDLC_%%V="

if not defined DRIVEDIT_LOCAL_CONFIG goto :ddlc_look
set "DD_LOCAL_CONFIG=%DRIVEDIT_LOCAL_CONFIG%"
if exist "%DD_LOCAL_CONFIG%" goto :ddlc_read
set "DD_ERR=DRIVEDIT_LOCAL_CONFIG names %DD_LOCAL_CONFIG%, which does not exist. Nothing was started."
exit /b 1

:ddlc_look
for %%I in ("%~dp0..\..") do set "DDLC_ROOT=%%~fI"
if exist "%DDLC_ROOT%\config\drivedit.local.env" (
  set "DD_LOCAL_CONFIG=%DDLC_ROOT%\config\drivedit.local.env"
  goto :ddlc_read
)
if exist "%DDLC_ROOT%\.git" if exist "%DDLC_ROOT%\drivedit.local.env" (
  set "DD_LOCAL_CONFIG=%DDLC_ROOT%\drivedit.local.env"
  goto :ddlc_read
)
set "DD_ERR=No local config: %DDLC_ROOT%\config\drivedit.local.env does not exist and DRIVEDIT_LOCAL_CONFIG is not set. This machine's paths live in that untracked file (see drivedit.local.env.example and tools\migrate_local_config.py). Nothing was started."
exit /b 1

:ddlc_read
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%DD_LOCAL_CONFIG%") do set "DDLC_%%A=%%B"
set "DDLC_K=WSL_DISTRO"
if not defined DDLC_WSL_DISTRO goto :ddlc_missing
set DDLC_WSL_DISTRO| findstr /R /X /C:"DDLC_WSL_DISTRO=[-0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz]*" >nul || goto :ddlc_badchars
set "DDLC_K=DATA_ROOT"
if not defined DDLC_DATA_ROOT goto :ddlc_missing
set DDLC_DATA_ROOT| findstr /R /X /C:"DDLC_DATA_ROOT=[-0123456789./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_DATA_ROOT%"
call :ddlc_towin || exit /b 1
set "DD_DATA_ROOT=%DDLC_DATA_ROOT%"
set "DD_DATA_ROOT_WIN=%DDLC_W%"
set "DD_DATA_DRIVE_WIN=%DDLC_U%:"
set "DDLC_K=GAME_INSTALL"
if not defined DDLC_GAME_INSTALL goto :ddlc_missing
set DDLC_GAME_INSTALL| findstr /R /X /C:"DDLC_GAME_INSTALL=[-0123456789 (),+./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_GAME_INSTALL%"
call :ddlc_towin || exit /b 1
set "DD_GAME_INSTALL_WIN=%DDLC_W%"
set "DDLC_K=TOOLS_INSTALL"
if not defined DDLC_TOOLS_INSTALL goto :ddlc_missing
set DDLC_TOOLS_INSTALL| findstr /R /X /C:"DDLC_TOOLS_INSTALL=[-0123456789 (),+./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_TOOLS_INSTALL%"
call :ddlc_towin || exit /b 1
set "DD_TOOLS_INSTALL_WIN=%DDLC_W%"
set "DD_WSL_DISTRO=%DDLC_WSL_DISTRO%"
set "DRIVEDIT_LOCAL_CONFIG=%DD_LOCAL_CONFIG%"
if defined WSLENV (set "WSLENV=%WSLENV%:DRIVEDIT_LOCAL_CONFIG/p") else set "WSLENV=DRIVEDIT_LOCAL_CONFIG/p"
if /I "%~1"=="/print" set DD_
exit /b 0

:ddlc_towin
rem DDLC_P is a WSL path /mnt/x/rest: DDLC_U becomes X and DDLC_W X:\rest, every / a \.
rem Refused: no /mnt/x/ prefix with a lower-case drive letter x, nothing after it, or a trailing /.
set "DDLC_U="
set "DDLC_R="
if not "%DDLC_P:~0,5%"=="/mnt/" goto :ddlc_badpath
set "DDLC_L=%DDLC_P:~5,1%"
set "DDLC_R=%DDLC_P:~6%"
if not defined DDLC_R goto :ddlc_badpath
if not "%DDLC_R:~0,1%"=="/" goto :ddlc_badpath
if "%DDLC_R:~-1%"=="/" goto :ddlc_badpath
for %%U in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do if /I "%DDLC_L%"=="%%U" set "DDLC_U=%%U"
if not defined DDLC_U goto :ddlc_badpath
if "%DDLC_L%"=="%DDLC_U%" goto :ddlc_badpath
set "DDLC_W=%DDLC_U%:%DDLC_R:/=\%"
exit /b 0

:ddlc_badpath
set "DD_ERR=Local config %DD_LOCAL_CONFIG%: %DDLC_K% is not a WSL path on a Windows drive (/mnt/x/... with a lower-case drive letter x, no trailing /). Nothing was started."
exit /b 1

:ddlc_badchars
set "DD_ERR=Local config %DD_LOCAL_CONFIG%: %DDLC_K% holds a character the launchers do not accept (run tools/local_config.py check). Nothing was started."
exit /b 1

:ddlc_missing
set "DD_ERR=Local config %DD_LOCAL_CONFIG% has no %DDLC_K%. Add a line %DDLC_K%=... (drivedit.local.env.example lists every key). Nothing was started."
exit /b 1
