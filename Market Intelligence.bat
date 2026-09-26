@echo off
rem ---------------------------------------------------------------
rem  Market Intelligence - double-click launcher
rem
rem  Starts the local API and opens the console UI in the browser.
rem  Close the console window to stop the app.
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo Starting Market Intelligence...
%PYTHON% app.py
if errorlevel 1 (
    echo.
    echo The app failed to start. Press any key to close.
    pause >nul
)
endlocal
