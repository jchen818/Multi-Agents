# Shared by scripts/gate and scripts/pre-push.
set -uo pipefail
root=$(git rev-parse --show-toplevel); cd "$root"
STAMPS="$root/state/gate"
AGENT_RE='^(agents/|core/|tools/|evals/lab_ops/)'
PLANNER_RE='^(agents/test_planner/|core/(recipes|mcp|guard|models)\.py|tools/agfhc/|config/plan_templates/|sim/|evals/planner/)'
# Files changed on this branch compared with origin/main (what a push to main brings in)
changed_files() { git fetch -q origin main 2>/dev/null; git diff --name-only "$(git merge-base HEAD origin/main)" HEAD; }
needs_evals() { changed_files | grep -qE "$AGENT_RE|$PLANNER_RE"; }
