# The feature cache is not numerically neutral — and it looks like a bug fix

**Status: open. Must be resolved before the 3-seed numbers go in the paper.**

## What happened

I added a frozen-encoder feature cache to make 3 seeds affordable, and wrote in
the commit message that it "changes nothing scientifically: identical inputs,
identical mask, identical frozen weights."

That claim is wrong. Same condition, same seed, same step:

| run | L0_native, seed 1337, step 1000 | val PER |
|---|---|---:|
| original (no cache) | `L0_native step 1000` | **0.5314** |
| cached | `L0_native_s1337 step 1000` | **0.4530** |

A 0.078 PER difference — roughly 8× the ~0.01 resolution floor the single-seed
run implied. Something real changed.

## Two candidate causes

**1. fp16 storage.** The cache stores WavLM layer-9 activations as fp16 and
casts back to fp32 on use. That loses precision. It should hurt, not help, so
it does not explain an *improvement*.

**2. Zero-pad contamination — the likely one.** In the original path, a batch of
audio was padded to the batch maximum and the whole padded hidden-state tensor
was handed to the head, with a key-padding mask. That mask is respected by
attention — but **not** by the depthwise convolution or the feed-forward in
`MaskedBlock`. So the causal conv could smear padded positions into real frames
near the end of every short utterance in a batch. CTC itself was fine, because
it used the true `in_len`.

The cache stores `h[j, :fl[j]]` — each utterance trimmed to its true length —
and re-pads at collate time. Same masking, but the padded region no longer
contains encoder output derived from zero-padded audio.

If that is the cause, the cached path is **more correct**, and the original
single-seed numbers carried a padding artefact whose size depended on batch
composition. That would also help explain the one non-monotone step in the
original transcription curve.

## What has to happen before publishing

1. **Isolate it.** Re-run one condition three ways: (a) original path,
   (b) cache in fp32, (c) cache in fp16. If (b) ≈ (c) ≠ (a), it is padding, not
   precision.
2. **Fix the conv/FFN padding** in `MaskedBlock` regardless — zero the padded
   positions before the depthwise conv. It is a real bug whether or not it
   explains this gap.
3. **Do not mix the two runs.** The 1-seed table in `RESULTS_M4.md` F11 and
   `paper/…draft.md` §5.2 came from the uncached path. Either re-run F11 under
   the cached path or label both clearly. They are not the same experiment.

## What this does *not* threaten

The H3 direction and the monotone canonical-preference growth are *within-run*
comparisons — every condition in a given run shared the same code path, so a
systematic padding artefact affects the level, not the ordering. But the
magnitudes (1.48×, 2.8×) must be re-quoted from whichever run the paper uses.
