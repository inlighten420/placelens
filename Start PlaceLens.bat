@echo off
rem PlaceLens Studio: double-click to start. The first time it sets itself up (a few minutes), then opens your browser.
setlocal
cd /d "%~dp0"
title PlaceLens Studio

if exist ".venv\Scripts\placelens.exe" goto run

echo Setting up PlaceLens (first time only, a few minutes)...
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
  echo.
  echo PlaceLens needs Python 3.10 or newer, and it isn't on this computer.
  echo Opening the download page: install it, tick "Add python.exe to PATH", then double-click this again.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)
%PY% -m venv .venv || goto failed
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -e . || goto failed

:run
echo Starting PlaceLens Studio. Keep this window open while you use it; close it to stop.
".venv\Scripts\placelens.exe" studio
pause
exit /b 0

:failed
echo.
echo Setting up didn't work (see above). Check your internet connection and try again.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1
