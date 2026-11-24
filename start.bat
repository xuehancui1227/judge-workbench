@echo off
setlocal
cd /d "%~dp0"
python "%~dp0launcher.py" %*
if errorlevel 1 pause
endlocal
