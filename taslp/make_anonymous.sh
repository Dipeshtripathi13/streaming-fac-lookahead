#!/bin/bash
# Build a double-blind version of the manuscript.
#
# Only run this if TASLP requires anonymised submission. Confirm on the
# Information for Authors page first; IEEE Signal Processing Society
# transactions have not historically been double-blind, but do not submit on
# my assumption, and do not submit the identified version if they are.
#
# What identifies the author here is not just the name. The repository URL
# is github.com/Dipeshtripathi13/..., which names him, and it appears twice:
# in the title-page footnote and in the pre-specification footnote in §V-E.
# Removing the name and leaving those in place would defeat the exercise.
#
#   ./make_anonymous.sh  ->  anonymous/manuscript_anon.pdf
set -euo pipefail
cd "$(dirname "$0")"
OUT=anonymous
rm -rf "$OUT" && mkdir -p "$OUT"
cp source/taslp.bbl source/refs.bib source/fig_dense_no_knee.png "$OUT/"

python3 - <<'PY'
import re
s = open('source/taslp.tex').read()

# 1. author block. Both \thanks footnotes go with it, but the second one also
#    carried the data and licence statement, which is not identifying and is
#    worth keeping for review, so it is re-added without the repository link.
s = re.sub(r'\\author\{.*?\}\}\n',
           '\\\\author{Anonymous Author(s)%\\n'
           '\\\\thanks{Code and per-condition outputs are public; the repository is '
           'withheld for review. Data is L2-ARCTIC~\\\\protect\\\\cite{l2arctic2018} via '
           '\\\\texttt{KoelLabs/L2Arctic}. Given that corpus\'s CC-BY-NC-4.0 licence, '
           'we conservatively release configurations rather than model weights.}}\n',
           s, flags=re.S)

# 2. the repository URL, wherever it appears
s = s.replace(r'\url{https://github.com/Dipeshtripathi13/streaming-fac-lookahead}',
              '[repository URL withheld for review]')
s = re.sub(r'\\footnote\{\\url\{https://github\.com/Dipeshtripathi13[^}]*\}\}',
           r'\\footnote{[repository URL withheld for review]}', s)
s = re.sub(r'https://github\.com/Dipeshtripathi13[^\s}]*',
           '[repository URL withheld for review]', s)

# 3. self-identifying phrasing in the acknowledgements
s = s.replace('made a single-author study of this scope\npossible',
              'made a study of this scope possible')
s = s.replace('The author declares no competing interests.',
              'The author(s) declare no competing interests.')

open('anonymous/taslp_anon.tex', 'w').write(s)
print('  anonymised source written')
PY

( cd "$OUT" && tectonic taslp_anon.tex >/dev/null 2>&1 ) \
  || { echo "FAIL: anonymised source does not compile"; exit 1; }

# refuse to ship anything that still names the author
LEAK=$(python3 - <<'PY'
import pypdf, re, sys
t = "".join((p.extract_text() or "") for p in pypdf.PdfReader('anonymous/taslp_anon.pdf').pages)
bad = [w for w in ('Tripathi', 'Dipeshtripathi13', 'coyotes.usd.edu',
                   'South Dakota', 'dipesh') if re.search(w, t, re.I)]
print(','.join(bad))
PY
)
if [ -n "$LEAK" ]; then
  echo "FAIL: identifying strings still in the PDF: $LEAK"; exit 1
fi
mv "$OUT"/taslp_anon.pdf "$OUT"/manuscript_anon.pdf
echo "wrote $OUT/manuscript_anon.pdf ($(qpdf --show-npages "$OUT"/manuscript_anon.pdf) pages, no identifying strings)"
