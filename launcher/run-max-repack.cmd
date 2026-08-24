@echo off
rem Thin wrapper: all logic lives in run-local-moe.ps1.
rem Usage: run-max-repack.cmd -Model <path to .gguf> [-Server] [-Port N] [any other run-local-moe.ps1 switch]
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-local-moe.ps1" -Profile MAX_SPEED_REPACK %*
exit /b %ERRORLEVEL%
