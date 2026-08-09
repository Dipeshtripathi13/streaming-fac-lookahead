#!/bin/bash
# Commit everything, clearing the stale lock files first.
#
# Why this exists
# ---------------
# The assistant works through a sandbox mount that permits writes but DENIES
# unlink inside .git. Git creates a lock file for every ref/index update and
# removes it on success -- so every git write from the sandbox succeeds once and
# leaves an orphaned lock that blocks the next one. Pointing GIT_INDEX_FILE
# outside the mount fixes .git/index.lock, but a commit must also lock HEAD and
# refs/heads/<branch>, and those locks have to live in .git.
#
# The locks are always zero-byte orphans in this situation, so this script only
# deletes ones that are empty AND older than 60 seconds -- it will not stomp on a
# lock held by a genuinely running git process (e.g. your own editor).
#
# Usage:
#   ./gitsave.sh                       # commit with a generated message
#   ./gitsave.sh "my message"          # commit with that message
#   ./gitsave.sh -F path/to/msg.txt    # commit with a message file
set -uo pipefail
cd "$(dirname "$0")" || exit 1

echo "== clearing stale zero-byte locks older than 60s =="
CLEARED=0
while IFS= read -r lock; do
  # only if empty and not freshly created
  if [ ! -s "$lock" ]; then
    if [ -z "$(find "$lock" -newermt '-60 seconds' 2>/dev/null)" ]; then
      rm -f "$lock" && { echo "  removed $lock"; CLEARED=$((CLEARED+1)); }
    else
      echo "  SKIP (fresh, may be in use): $lock"
    fi
  else
    echo "  SKIP (non-empty, real lock): $lock"
  fi
done < <(find .git -name '*.lock' 2>/dev/null)
find .git/objects -name 'tmp_obj_*' -delete 2>/dev/null
echo "  cleared $CLEARED lock(s)"

# Rebuild the index from HEAD before staging. The assistant commits using
# GIT_INDEX_FILE pointed outside the mount (to dodge index.lock), which means
# .git/index is left at its pre-commit state and reports already-committed files
# as deleted. read-tree resyncs it; git add -A then stages the real worktree.
echo
echo "== resyncing index with HEAD =="
git read-tree HEAD 2>/dev/null && echo "  index rebuilt from HEAD" \
  || echo "  read-tree skipped (no HEAD yet?)"

if [ -n "$(git status --porcelain)" ]; then
  echo
  echo "== staging =="
  git add -A || { echo "git add failed"; exit 1; }
  N=$(git diff --cached --name-only | wc -l | tr -d ' ')
  echo "  $N file(s) staged"

  echo
  echo "== committing =="
  if [ "${1:-}" = "-F" ] && [ -n "${2:-}" ]; then
    git commit -F "$2" || exit 1
  elif [ -n "${1:-}" ]; then
    git commit -m "$1" || exit 1
  else
    git commit -m "Work in progress: $(date +%Y-%m-%d\ %H:%M)" || exit 1
  fi
else
  echo
  echo "Nothing to commit."
fi

echo
git log --oneline -3
git status --short --branch | head -1
echo
echo "Not pushed. To publish:  git push origin main"
