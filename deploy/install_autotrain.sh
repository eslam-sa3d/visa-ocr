#!/bin/sh
# Installs the weekly retraining timer on the training server. Run as root from the project folder:
#   sudo sh deploy/install_autotrain.sh
# Safe to run again. It does not start a training run; the timer does, on Sunday 02:00.
set -eu
[ "$(id -u)" = "0" ] || { echo "Run as root (sudo)." >&2; exit 1; }
command -v systemctl >/dev/null || { echo "systemd not found: use the cron line in docs/MODEL_TRAINING.md instead." >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
id visaocr >/dev/null 2>&1 || useradd --system --home /opt/visa-ocr --shell /usr/sbin/nologin visaocr
mkdir -p /etc/visa-ocr
if [ ! -f /etc/visa-ocr/autotrain.env ]; then
  install -m 600 "$HERE/autotrain.env.example" /etc/visa-ocr/autotrain.env
  echo "Created /etc/visa-ocr/autotrain.env from the example: edit the paths before the first run."
fi
install -m 644 "$HERE/visa-ocr-autotrain.service" "$HERE/visa-ocr-autotrain.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now visa-ocr-autotrain.timer
systemctl list-timers visa-ocr-autotrain.timer --no-pager || true
echo "Installed. Try one run now:  systemctl start visa-ocr-autotrain.service && journalctl -u visa-ocr-autotrain.service -n 50"
