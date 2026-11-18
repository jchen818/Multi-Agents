#!/usr/bin/env bash
# Install / update the agents' systemd services on nfs-server (Phase 6.4).
#   deploy/systemd/install.sh            # install units, enable + (re)start 1 triage worker and 1 coder worker
#   deploy/systemd/install.sh 2          # run 2 triage workers (the coder worker is always 1)
#   deploy/systemd/install.sh --remove   # stop, disable and remove the units
# Needs sudo for /etc/systemd/system. The services run as the user who runs this script, from this checkout.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
USER_NAME=$(id -un); GROUP_NAME=$(id -gn)
UNITS="agents-triage-worker@.service agents-coder-worker@.service"

if [ "${1:-}" = "--remove" ]; then
  for u in $(systemctl list-units --all --plain --no-legend 'agents-triage-worker@*' 'agents-coder-worker@*' | awk '{print $1}'); do
    sudo systemctl disable --now "$u"
  done
  for unit in $UNITS; do sudo rm -f "/etc/systemd/system/$unit"; done
  sudo systemctl daemon-reload
  echo "removed $UNITS"; exit 0
fi

N=${1:-1}
[[ "$N" =~ ^[1-9]$ ]] || { echo "worker count must be 1-9"; exit 1; }
[ -f "$ROOT/agents.env" ] || { echo "missing $ROOT/agents.env"; exit 1; }
grep -q '^REDIS_URL=' "$ROOT/agents.env" || { echo "agents.env has no REDIS_URL (see deploy/redis)"; exit 1; }
[ -x "$ROOT/.venv/bin/python" ] || { echo "missing $ROOT/.venv"; exit 1; }
if [ "$(stat -c %a "$ROOT/agents.env")" != "600" ]; then
  echo "note: agents.env holds secrets; tightening it to 600"; chmod 600 "$ROOT/agents.env"
fi

for unit in $UNITS; do
  sed -e "s#@ROOT@#$ROOT#g" -e "s#@USER@#$USER_NAME#g" -e "s#@GROUP@#$GROUP_NAME#g" \
      "$ROOT/deploy/systemd/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
# the coder worker prepares gpu-validation changes (never pushes); it needs the GitHub token and Docker
if [ -f "$ROOT/secrets/github.yaml" ] && docker info >/dev/null 2>&1; then
  sudo systemctl enable agents-coder-worker@1 >/dev/null
  sudo systemctl restart agents-coder-worker@1
else
  echo "coder worker not started: needs secrets/github.yaml and Docker access for $USER_NAME"
fi
for i in $(seq 1 "$N"); do
  sudo systemctl enable "agents-triage-worker@$i" >/dev/null
  sudo systemctl restart "agents-triage-worker@$i"
done
# stop instances above N
for u in $(systemctl list-units --all --plain --no-legend 'agents-triage-worker@*' | awk '{print $1}'); do
  i=${u#agents-triage-worker@}; i=${i%.service}
  if [[ "$i" =~ ^[0-9]+$ ]] && [ "$i" -gt "$N" ]; then sudo systemctl disable --now "$u"; fi
done
sleep 3
systemctl --no-pager --lines=0 status 'agents-triage-worker@*' 'agents-coder-worker@*' | grep -E '●|Active:' || true
echo "logs: journalctl -u 'agents-triage-worker@*' -u 'agents-coder-worker@*' -f"
