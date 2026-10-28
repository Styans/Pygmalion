#!/bin/sh
# Запуск сервера DroneGen, установленного командой: python3 install.py
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/runtime/start_server.sh" ]; then
  echo "DroneGen ещё не установлен. Сначала выполните: python3 install.py" >&2
  exit 1
fi
exec "$DIR/runtime/start_server.sh" "$@"
