@echo off
rem ==========================================================================================
rem DriveDiT local paths for the Windows launchers (ticket DD-22). Every launcher in this folder
rem CALLs it right after setlocal DisableDelayedExpansion:   call "%~dp0dd_local_config.cmd"
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
rem The Windows form is the pure mapping of tools/local_config.py: /mnt/x/... becomes X:\....
rem It also sets DRIVEDIT_LOCAL_CONFIG to the file it read and adds DRIVEDIT_LOCAL_CONFIG/p to
rem WSLENV, so the preflight in WSL reads the very same file.
rem
rem Nothing is set, created or started unless all of these hold, checked in this order:
rem   1. The file has the grammar of tools/local_config.py: every line is a comment (#...), empty
rem      (a line of blanks or tabs is refused), or KEY=VALUE with KEY one of its nine keys, each key
rem      at most once, and a VALUE of letters, digits, space and _ . , + ( ) ~ / - only (so no line
rem      can start with =, hold a second =, or carry a tab or a cmd.exe metacharacter). It is read
rem      with delayed expansion off, so ! stays a character.
rem   2. The four keys used here are as tools/local_config.py requires: WSL_DISTRO [A-Za-z0-9._-],
rem      DATA_ROOT /mnt/x/[A-Za-z0-9_./-], the installs /mnt/x/..., each a normalised path (no
rem      trailing /, no //, no . or .. component, no white space at either end).
rem   3. The data drive is not the Windows system drive: not %SystemDrive% (which no config sets),
rem      and it holds no Windows\System32 folder. A launcher calls this helper before it creates,
rem      copies or starts anything, so a system-drive DATA_ROOT stops every launcher first.
rem   4. The Python reading of the same file accepts it too: tools\local_config.py validate, next to
rem      tools\launch, run in WSL the way the launchers run the preflight (the whole grammar and
rem      every key, the ones not used here included). Any difference stops here.
rem Otherwise it exits with code 1 and DD_ERR says why.
rem It reads one file and writes nothing. "dd_local_config.cmd /print" prints the DD_ values
rem instead of returning them (tools/local_config.py check --cmd compares them with Python's).
rem ==========================================================================================
setlocal DisableDelayedExpansion
set "ERRORLEVEL="
set "DD_ERR="
set "DD_LOCAL_CONFIG="
set "DDLC_KEYS=DATA_ROOT REALIGN_ROOT CODE_ROOT GAME_INSTALL TOOLS_INSTALL EXPERIMENTAL_GAME_INSTALL EXPERIMENTAL_TOOLS_INSTALL SYSTEM_DRIVE WSL_DISTRO"
for %%V in (%DDLC_KEYS% K P L U R W ROOT TOOLS DUP) do set "DDLC_%%V="

if not defined DRIVEDIT_LOCAL_CONFIG goto :ddlc_look
set "DD_LOCAL_CONFIG=%DRIVEDIT_LOCAL_CONFIG%"
if exist "%DD_LOCAL_CONFIG%" goto :ddlc_read
set "DD_ERR=DRIVEDIT_LOCAL_CONFIG names %DD_LOCAL_CONFIG%, which does not exist. Nothing was started."
goto :ddlc_fail

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
goto :ddlc_fail

:ddlc_read
rem 1. The grammar, before any value is read (findstr also reads files with LF line ends; its /X does not):
rem    a line that is not a comment, is not empty (blanks or tabs only refuse too, as in Python) and does not
rem    start KEY= plus a non-= ...
findstr /R /V /B /C:"#" /C:"DATA_ROOT=[^=]" /C:"REALIGN_ROOT=[^=]" /C:"CODE_ROOT=[^=]" /C:"GAME_INSTALL=[^=]" /C:"TOOLS_INSTALL=[^=]" /C:"EXPERIMENTAL_GAME_INSTALL=[^=]" /C:"EXPERIMENTAL_TOOLS_INSTALL=[^=]" /C:"SYSTEM_DRIVE=[^=]" /C:"WSL_DISTRO=[^=]" "%DD_LOCAL_CONFIG%" | findstr /R /C:"." >nul && goto :ddlc_grammar
rem    ... a second = in a line that is not a comment, or a character outside the value characters
findstr /R /V /B /C:"#" "%DD_LOCAL_CONFIG%" | findstr /R /C:"=.*=" >nul && goto :ddlc_grammar
findstr /R /V /B /C:"#" "%DD_LOCAL_CONFIG%" | findstr /R /C:"[^-0123456789 (),+./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~=]" >nul && goto :ddlc_grammar
for %%K in (%DDLC_KEYS%) do for /f %%N in ('findstr /B /C:"%%K=" "%DD_LOCAL_CONFIG%" ^| find /C /V ""') do if %%N GTR 1 set "DDLC_DUP=%%K"
if defined DDLC_DUP goto :ddlc_twice
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%DD_LOCAL_CONFIG%") do set "DDLC_%%A=%%B"
rem 2. The four keys used here.
set "DDLC_K=WSL_DISTRO"
if not defined DDLC_WSL_DISTRO goto :ddlc_missing
set DDLC_WSL_DISTRO| findstr /R /X /C:"DDLC_WSL_DISTRO=[-0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz]*" >nul || goto :ddlc_badchars
set "DDLC_K=DATA_ROOT"
if not defined DDLC_DATA_ROOT goto :ddlc_missing
set DDLC_DATA_ROOT| findstr /R /X /C:"DDLC_DATA_ROOT=[-0123456789./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_DATA_ROOT%"
call :ddlc_towin || goto :ddlc_fail
set "DD_DATA_ROOT=%DDLC_DATA_ROOT%"
set "DD_DATA_ROOT_WIN=%DDLC_W%"
set "DD_DATA_DRIVE_WIN=%DDLC_U%:"
set "DDLC_K=GAME_INSTALL"
if not defined DDLC_GAME_INSTALL goto :ddlc_missing
set DDLC_GAME_INSTALL| findstr /R /X /C:"DDLC_GAME_INSTALL=[-0123456789 (),+./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_GAME_INSTALL%"
call :ddlc_towin || goto :ddlc_fail
set "DD_GAME_INSTALL_WIN=%DDLC_W%"
set "DDLC_K=TOOLS_INSTALL"
if not defined DDLC_TOOLS_INSTALL goto :ddlc_missing
set DDLC_TOOLS_INSTALL| findstr /R /X /C:"DDLC_TOOLS_INSTALL=[-0123456789 (),+./ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~]*" >nul || goto :ddlc_badchars
set "DDLC_P=%DDLC_TOOLS_INSTALL%"
call :ddlc_towin || goto :ddlc_fail
set "DD_TOOLS_INSTALL_WIN=%DDLC_W%"
set "DD_WSL_DISTRO=%DDLC_WSL_DISTRO%"
rem 3. Never the Windows system drive.
if not defined SystemDrive goto :ddlc_nosystem
if /I "%DD_DATA_DRIVE_WIN%"=="%SystemDrive%" goto :ddlc_system
if exist "%DD_DATA_DRIVE_WIN%\Windows\System32\" goto :ddlc_system
rem 4. The Python reading of the same file, in WSL (it reads the file only).
set "DRIVEDIT_LOCAL_CONFIG=%DD_LOCAL_CONFIG%"
if defined WSLENV (set "DDLC_ENV=%WSLENV%:DRIVEDIT_LOCAL_CONFIG/p") else set "DDLC_ENV=DRIVEDIT_LOCAL_CONFIG/p"
for %%I in ("%~dp0..") do set "DDLC_TOOLS=%%~fI"
if not exist "%DDLC_TOOLS%\local_config.py" goto :ddlc_noloader
set "DDLC_LOADER=%DDLC_TOOLS%\local_config.py"
set "WSLENV=%DDLC_ENV%:DDLC_LOADER/p"
wsl.exe -d %DD_WSL_DISTRO% -e /usr/bin/python3 -I -B -c "import os, runpy, sys; sys.argv = ['local_config.py', 'validate']; runpy.run_path(os.environ['DDLC_LOADER'], run_name='__main__')" >nul
rem wsl.exe exits with a negative code when it cannot start the distro: only exactly 0 passes. The exit code is
rem tested with "if errorlevel 1" (1 or more) and "if not errorlevel 0" (below 0), never through %ERRORLEVEL%,
rem which an inherited variable of that name would shadow.
if errorlevel 1 goto :ddlc_python
if not errorlevel 0 goto :ddlc_python
set "WSLENV=%DDLC_ENV%"
if /I "%~1"=="/print" set DD_
endlocal & set "DD_LOCAL_CONFIG=%DD_LOCAL_CONFIG%" & set "DD_WSL_DISTRO=%DD_WSL_DISTRO%" & set "DD_DATA_ROOT=%DD_DATA_ROOT%" & set "DD_DATA_ROOT_WIN=%DD_DATA_ROOT_WIN%" & set "DD_DATA_DRIVE_WIN=%DD_DATA_DRIVE_WIN%" & set "DD_GAME_INSTALL_WIN=%DD_GAME_INSTALL_WIN%" & set "DD_TOOLS_INSTALL_WIN=%DD_TOOLS_INSTALL_WIN%" & set "DRIVEDIT_LOCAL_CONFIG=%DD_LOCAL_CONFIG%" & set "WSLENV=%WSLENV%" & set "DD_ERR="
exit /b 0

:ddlc_towin
rem DDLC_P is a WSL path /mnt/x/...: DDLC_U becomes X and DDLC_W X:\..., every / a \.
rem Refused: no /mnt/x/ prefix with a lower-case drive letter x, nothing after it, a trailing /,
rem white space at the end, or a component that is empty, . or .. (not a normalised path).
set "DDLC_U="
set "DDLC_R="
if not "%DDLC_P:~0,5%"=="/mnt/" goto :ddlc_badpath
set "DDLC_L=%DDLC_P:~5,1%"
set "DDLC_R=%DDLC_P:~6%"
if not defined DDLC_R goto :ddlc_badpath
if not "%DDLC_R:~0,1%"=="/" goto :ddlc_badpath
if "%DDLC_R:~-1%"=="/" goto :ddlc_badpath
if "%DDLC_R:~-1%"==" " goto :ddlc_badpath
if "%DDLC_R:~-2%"=="/." goto :ddlc_badpath
if "%DDLC_R:~-3%"=="/.." goto :ddlc_badpath
if not "%DDLC_R://=%"=="%DDLC_R%" goto :ddlc_badpath
if not "%DDLC_R:/./=%"=="%DDLC_R%" goto :ddlc_badpath
if not "%DDLC_R:/../=%"=="%DDLC_R%" goto :ddlc_badpath
for %%U in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do if /I "%DDLC_L%"=="%%U" set "DDLC_U=%%U"
if not defined DDLC_U goto :ddlc_badpath
if "%DDLC_L%"=="%DDLC_U%" goto :ddlc_badpath
set "DDLC_W=%DDLC_U%:%DDLC_R:/=\%"
exit /b 0

:ddlc_badpath
set "DD_ERR=Local config %DD_LOCAL_CONFIG%: %DDLC_K% is not a normalised WSL path on a Windows drive (/mnt/x/... with a lower-case drive letter x; no trailing /, no //, no . or .. component). Nothing was started."
exit /b 1

:ddlc_grammar
set "DD_ERR=Local config %DD_LOCAL_CONFIG% has a line that is not a comment, an empty line or KEY=VALUE with a known key and a plain value (run tools/local_config.py check). Nothing was started."
goto :ddlc_fail

:ddlc_twice
set "DD_ERR=Local config %DD_LOCAL_CONFIG% sets %DDLC_DUP% more than once (a key may appear once). Nothing was started."
goto :ddlc_fail

:ddlc_badchars
set "DD_ERR=Local config %DD_LOCAL_CONFIG%: %DDLC_K% holds a character the launchers do not accept (run tools/local_config.py check). Nothing was started."
goto :ddlc_fail

:ddlc_missing
set "DD_ERR=Local config %DD_LOCAL_CONFIG% has no %DDLC_K%. Add a line %DDLC_K%=... (drivedit.local.env.example lists every key). Nothing was started."
goto :ddlc_fail

:ddlc_nosystem
set "DD_ERR=SystemDrive is not set, so the Windows system drive cannot be told apart from the data drive. Nothing was started."
goto :ddlc_fail

:ddlc_system
set "DD_ERR=DATA_ROOT %DD_DATA_ROOT_WIN% of %DD_LOCAL_CONFIG% is on the Windows system drive (%SystemDrive%, or a drive that holds Windows\System32); nothing is ever written there. Nothing was started."
goto :ddlc_fail

:ddlc_noloader
set "DD_ERR=Missing %DDLC_TOOLS%\local_config.py: the launchers check the local config with it (install it with the tools). Nothing was started."
goto :ddlc_fail

:ddlc_python
set "DD_ERR=The Python check of %DD_LOCAL_CONFIG% (tools/local_config.py validate, in WSL %DD_WSL_DISTRO%) refused it; its message is above. Nothing was started."
goto :ddlc_fail

:ddlc_fail
endlocal & set "DD_ERR=%DD_ERR%"
exit /b 1
