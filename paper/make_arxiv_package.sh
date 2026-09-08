#!/bin/bash
# Build the arXiv source tarball for the TASLP paper.
#   ./make_arxiv_package.sh  ->  arxiv_taslp.tar.gz
#
# arXiv compiles the source itself, so the tarball must be self-sufficient.
# Two things that silently break it and are checked below:
#   * a missing .bbl -- arXiv does not reliably run BibTeX, and the paper
#     renders with every citation as [?]
#   * refs.bib is a symlink into the parent directory, so it must be
#     dereferenced (cp -L) or it arrives as a broken link
set -euo pipefail
cd "$(dirname "$0")"
SRC=taslp
OUT=arxiv_build
rm -rf "$OUT" && mkdir -p "$OUT"

( cd "$SRC" && tectonic --keep-intermediates taslp.tex >/dev/null 2>&1 )
[ -s "$SRC/taslp.bbl" ] || { echo "FAIL: no taslp.bbl produced"; exit 1; }

cp    "$SRC/taslp.tex" "$SRC/taslp.bbl" "$SRC/fig_dense_no_knee.png" "$OUT/"
cp -L "$SRC/refs.bib" "$OUT/"

# refuse to ship a package that does not build on its own
( cd "$OUT" && tectonic taslp.tex >/dev/null 2>&1 ) || { echo "FAIL: package does not compile standalone"; exit 1; }
PAGES=$(qpdf --show-npages "$OUT/taslp.pdf")
CITES=$(grep -c bibitem "$OUT/taslp.bbl")
# The [?] check is the one that actually catches a missing or stale .bbl, so
# it must not pass silently when pypdf is absent -- prefer the project venv.
PY_BIN=../.venv/bin/python3
[ -x "$PY_BIN" ] || PY_BIN=python3
"$PY_BIN" - "$OUT/taslp.pdf" <<'PYEOF'
import sys
try:
    import pypdf
except ImportError:
    sys.exit("FAIL: pypdf unavailable, cannot verify that citations resolved. "
             "Install it or run from the project venv rather than shipping an "
             "unchecked package.")
t = "".join((p.extract_text() or "") for p in pypdf.PdfReader(sys.argv[1]).pages)
n = t.count("[?]")
print(f"  unresolved citations in rendered PDF: {n}")
sys.exit(1 if n else 0)
PYEOF
rm -f "$OUT"/taslp.pdf "$OUT"/taslp.aux "$OUT"/taslp.log "$OUT"/taslp.blg
tar -czf arxiv_taslp.tar.gz -C "$OUT" .
echo "wrote arxiv_taslp.tar.gz ($PAGES pages, $CITES references)"
tar -tzf arxiv_taslp.tar.gz
