# Running the informal listening session

**This is not a rater study and must never be reported as one.** Its single
purpose is to catch the one failure that would embarrass the paper: the PER
ordering disagreeing with what is actually audible. If 3--5 people hear the
same ordering the curves predict, the paper is de-risked for free. If they do
not, that is something to find out now rather than from a reviewer.

## 1. Serve the page (it will NOT work by double-clicking)

`index.html` fetches `manifest.json`, and browsers block `fetch` over
`file://`. Serve it over HTTP instead:

```bash
cd results/listening_test
python3 -m http.server 8000
```

Then open <http://localhost:8000> . For someone on another machine on the same
network, give them `http://<your-ip>:8000`.

## 2. What each person does

* **Headphones, quiet room.** Say this explicitly; laptop speakers will not
  resolve the differences being judged.
* 30 forced-choice pairs, about 10 minutes.
* The instruction on the page is deliberately narrow: judge the *pronunciation
  of the sounds*, not audio quality, speed, or preference. Every clip is the
  same synthetic voice saying the same sentence, so anything else they might
  latch onto is noise.
* At the end they click **Download your responses** and send you the CSV.

Do not tell them what the conditions are, or that lookahead is the variable.
Filenames are hashed for exactly this reason.

## 3. Score it

Drop every returned CSV into `results/listening_test/votes/` and run:

```bash
python3 eval/listening_test.py score --votes results/listening_test/votes
```

The scorer accepts a directory, a glob, or a single file, and merges them.

It drops raters who fail the attention checks (ceiling-vs-floor pairs, where
the canonical-phone clip should obviously win) before fitting anything. A rater
who cannot pick the ceiling was not listening.

## 4. Reading the result honestly

* With 3--5 raters the Bradley--Terry confidence intervals will be wide and
  will overlap. **That is expected and is not a finding.** Do not report the
  scores as if they were a rater study.
* What you are looking for is the *ordering*: do the long-lookahead conditions
  tend to win over the short-lookahead ones, the way PER says they should?
* **If the ordering matches**, say nothing about it in the paper beyond what is
  already there. It was insurance, not a result.
* **If the ordering contradicts PER**, that is a genuine finding and it changes
  the paper: it would mean PER is improving on phones nobody can hear, which is
  precisely the objection §Limitations concedes. Tell the reader.

## 5. Caveats that apply no matter how it comes out

Every clip is TTS driven by a phone string, in one fixed voice. This tests
whether the *phone sequences* differ audibly. It says nothing about the
naturalness, prosody or speaker identity a real conversion system would have to
get right, and it is not a substitute for the funded study.
