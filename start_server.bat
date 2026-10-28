@echo off
chcp 65001 >nul
rem Запуск сервера DroneGen, установленного командой: python install.py
if not exist "%~dp0runtime\start_server.bat" (
  echo DroneGen ещё не установлен. Сначала выполните: python install.py
  pause
  exit /b 1
)
call "%~dp0runtime\start_server.bat" %*
