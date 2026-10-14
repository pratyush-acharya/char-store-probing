#!/usr/bin/env python3
"""Build a LARGE, clean, frequency-ranked English stimulus set for the character tasks, to replace
the ~100-word pool whose smallness let the answer probe memorize word->answer lookups.

Design goals:
  - thousands of words (so GroupKFold-by-word has real held-out generalization and no single word
    dominates), lowercase ascii a-z only, length 3-9 (keeps single-digit answers for word_length /
    count_distinct / counting stratification),
  - ranked by corpus frequency (common words -> clean tokenization / non-degenerate readout),
  - enough per-length coverage for word_length, and enough letter-count variety for counting.

Source: NLTK Brown frequency (common English); falls back to /usr/share/dict/words if Brown is
unavailable (no frequency -> alphabetical, still clean). Writes src/stimulus_words.txt (one word per
line, frequency-desc). Bundled in the repo (src/ is rsynced; data/ is excluded), so cloud runs need
no NLTK download.

Usage:  uv run python scripts/build_stimulus_words.py [--n 6000] [--min-len 3] [--max-len 9]
Pure text processing -- no model, no GPU.
"""
import argparse, re, sys
from pathlib import Path
from collections import Counter

OUT = Path(__file__).resolve().parent.parent / "src" / "stimulus_words.txt"
WORD_RE = re.compile(r"^[a-z]+$")


def from_brown():
    """(word, freq) for lowercase alphabetic Brown tokens, freq-desc. None if unavailable."""
    try:
        import nltk
        try:
            from nltk.corpus import brown
            brown.words()  # trigger LookupError if not downloaded
        except LookupError:
            nltk.download("brown", quiet=True)
            from nltk.corpus import brown
        fd = Counter(w.lower() for w in brown.words())
        return fd.most_common()
    except Exception as e:  # no nltk / no download / offline
        print(f"[brown unavailable: {e}]", file=sys.stderr)
        return None


def from_system_dict():
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        fp = Path(p)
        if fp.exists():
            words = [w.strip().lower() for w in fp.read_text(errors="ignore").splitlines()]
            return [(w, 1) for w in words]  # no frequency
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="target vocabulary size")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=9)
    ap.add_argument("--min-per-len", type=int, default=200,
                    help="ensure at least this many words at each length (for word_length balance)")
    args = ap.parse_args()

    ranked = from_brown()
    source = "brown-frequency"
    if not ranked:
        ranked = from_system_dict()
        source = "system-dict (alphabetical, no frequency)"
    if not ranked:
        print("ERROR: no word source (need nltk brown or /usr/share/dict/words)"); raise SystemExit(1)

    # filter: clean lowercase a-z, length window, dedup preserving freq order
    seen, kept = set(), []
    by_len = Counter()
    for w, _ in ranked:
        if w in seen or not WORD_RE.match(w):
            continue
        if not (args.min_len <= len(w) <= args.max_len):
            continue
        seen.add(w); kept.append(w); by_len[len(w)] += 1

    # take the top-n by frequency, but guarantee per-length coverage by topping up short lengths
    top = kept[: args.n]
    top_set = set(top)
    have = Counter(len(w) for w in top)
    for L in range(args.min_len, args.max_len + 1):
        if have[L] < args.min_per_len:
            for w in kept:
                if len(w) == L and w not in top_set:
                    top.append(w); top_set.add(w); have[L] += 1
                    if have[L] >= args.min_per_len:
                        break

    top = sorted(set(top), key=lambda w: (len(w), w))  # stable, length-then-alpha
    OUT.write_text("\n".join(top) + "\n")

    print(f"source: {source}")
    print(f"wrote {len(top)} words -> {OUT}")
    print("per-length counts:", {L: sum(1 for w in top if len(w) == L)
                                 for L in range(args.min_len, args.max_len + 1)})
    print("sample:", top[:8], "...", top[-4:])


if __name__ == "__main__":
    main()
