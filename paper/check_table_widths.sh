#!/bin/bash
# Measure every table against the column it must fit in.
#
# Why this exists: a `tabular` that is too wide for an IEEE column does NOT
# produce an "Overfull \hbox" warning when it sits centred inside a `table`
# float. It silently protrudes into the neighbouring column. Table IV shipped
# in a review draft at 312pt against a 252pt column -- 60pt of overlap that a
# clean tectonic run reported as zero errors and zero overfull boxes.
#
#   ./check_table_widths.sh [taslp/taslp.tex]
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-taslp/taslp.tex}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python3 - "$SRC" > "$WORK/probe.tex" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
pre = src[:src.index('\\begin{document}')]
out = []
for m in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', src, re.S):
    blk = m.group(0)
    star = blk.startswith('\\begin{table*}')
    tab = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', blk, re.S)
    if not tab:
        continue
    lab = re.search(r'\\label\{([^}]+)\}', blk)
    size = ''.join(re.findall(r'\\(footnotesize|small|scriptsize|tiny)\b', blk)[:1])
    tcs = re.search(r'\\setlength\{\\tabcolsep\}\{([^}]+)\}', blk)
    out.append(
        "\\begin{lrbox}{\\tb}" + (('\\' + size) if size else '')
        + (f'\\setlength{{\\tabcolsep}}{{{tcs.group(1)}}}' if tcs else '') + "%\n"
        + tab.group(0) + "\\end{lrbox}\n"
        + f"\\typeout{{TABW::{lab.group(1) if lab else 'unlabelled'}::"
          f"{'2col' if star else '1col'}::\\the\\wd\\tb}}")
print(pre)
print("\\begin{document}\n\\newsavebox{\\tb}")
print("\\typeout{TABW::__COL__::-::\\the\\columnwidth}")
print("\\typeout{TABW::__TXT__::-::\\the\\textwidth}")
print("\n".join(out))
print("x\n\\end{document}")
PY

cp "$(dirname "$SRC")"/*.png "$WORK/" 2>/dev/null || true
( cd "$WORK" && tectonic --print probe.tex 2>&1 ) | grep "TABW::" | sort -u | python3 -c "
import sys, re
col = txt = None; rows = []
for line in sys.stdin:
    m = re.search(r'TABW::([^:]+)::([^:]+)::([0-9.]+)pt', line)
    if not m: continue
    name, kind, w = m.group(1), m.group(2), float(m.group(3))
    if name == '__COL__': col = w
    elif name == '__TXT__': txt = w
    else: rows.append((name, kind, w))
bad = 0
print(f'column {col:.0f}pt, text {txt:.0f}pt')
for name, kind, w in rows:
    lim = col if kind == '1col' else txt
    ok = w <= lim
    bad += not ok
    print(f'  {name:<18} {kind} {w:7.1f}pt / {lim:.0f}pt  ' +
          ('ok' if ok else f'OVERFLOWS by {w-lim:.1f}pt'))
print(('\nFAIL: %d table(s) overflow' % bad) if bad else '\nall tables fit')
raise SystemExit(1 if bad else 0)
"
