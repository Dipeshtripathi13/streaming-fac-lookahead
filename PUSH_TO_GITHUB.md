# Pushing this repo to GitHub

The repo is already initialised and committed locally (1.3 MB, 75 files, no
corpora or checkpoints). Three steps left, all of which need your credentials —
which is why they're yours and not mine.

## 1. Clear the stale git locks

My sandbox couldn't delete its own lock files (filesystem permissions), so run:

```bash
cd ~/Desktop/accent_con/research
rm -f .git/*.lock .git/refs/heads/*.lock .git/objects/*.lock
git branch -M main
git log --oneline -1      # should show one commit
git status --short        # should be empty
```

## 2. Create the GitHub repo

At https://github.com/new — name it e.g. `streaming-fac-lookahead`.
**Public**, and do NOT let it add a README, .gitignore or licence (we have all
three; an auto-added file causes a merge conflict on first push).

## 3. Push

```bash
git remote add origin https://github.com/Dipeshtripathi13/streaming-fac-lookahead.git
git push -u origin main
```

Then paste the clone URL into cell 2 of `notebooks/colab_streaming_fac.ipynb`.

---

## What is and is not in the repo

**In:** all code, the four docs, requirements, the Colab notebook, the runner
scripts, and `results/` — raw CSV/JSONL plus figures. Results are committed on
purpose: they are small, and a benchmark repo whose numbers can't be traced to
a file is not reproducible.

**Out** (see `.gitignore`): `.venv/`, `data/corpora/`, any `.wav`/`.onnx`,
checkpoints, `results/raw/_superseded/`.

**Licence:** MIT for the code. The `LICENSE` file carries an explicit note that
it does **not** cover L2-ARCTIC (CC-BY-NC-4.0, non-commercial) or the
pretrained checkpoints. Worth reading before you release weights — the
non-commercial corpus is what constrains that decision.

## Before making it public, check

- [ ] No HF token anywhere: `git grep -iE "hf_[A-Za-z0-9]{20,}"` returns nothing
- [ ] No absolute home paths in code: `git grep -n "/Users/dipeshtripathi" -- '*.py' '*.sh'`
      (harmless in docs, wrong in code)
- [ ] The repo is genuinely public if you want Colab to `git clone` without auth
