@echo off
rem List the available profiles.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-local-moe.ps1" -List
exit /b %ERRORLEVEL%
