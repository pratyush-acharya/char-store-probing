#!/usr/bin/env python3
"""Drift check: compare the current paper source with a baseline git tag.

1. UNTRACED numbers: decimals in the current LaTeX (sections, tables, main.tex) that appear neither
   in the baseline LaTeX nor in the analysis outputs (outputs/v4/analysis/*.json,
   paper/tables/numbers.json) at any rounding. Every one needs a source.
2. DROPPED numbers: decimals in the baseline that no longer appear in the current LaTeX. Allowed
   (e.g. moved into a figure) but each should be a deliberate choice.
3. WORDING flags: phrases the claims ledger (paper/CLAIMS.md) rules out, in the current source.
4. Section inventory: section and paragraph headings added or removed.

Usage:  uv run python scripts/check_paper_drift.py --baseline <git ref of the earlier paper>
Exit code 1 if any UNTRACED number or WORDING flag is found.
"""
import argparse, glob, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![\w.])[-+−]?\d*\.\d+")
# phrases ruled out by CLAIMS.md ("Not claimed"); matched case-insensitively in the LaTeX source
FORBIDDEN = [
    r"\brout(e|es|ed|ing)\b", r"\bcomputes?\b", r"\bscaling law\b", r"\bproves?\b",
    r"\bcausal(ly)?\b(?! claim)", r"\bthe model uses\b", r"\buses the (store|late signal|span)\b",
    r"\bmore information\b", r"\bdemonstrates? that the model\b", r"\bunderstands?\b",
]
# a flagged phrase is allowed only if its own sentence negates or scopes it
ALLOW_CONTEXT = [r"causal attention", r"\bno causal\b", r"\bnot\b", r"\bnor\b", r"\bcannot\b",
                 r"\bavoid\b", r"\bwithout\b", r"\bnever\b", r"interventions that pass"]


def tex_files(rev=None):
    if rev is None:
        files = [ROOT / "paper/main.tex", *sorted((ROOT / "paper/sections").glob("*.tex")),
                 *sorted((ROOT / "paper/tables").glob("*.tex"))]
        return {str(f.relative_to(ROOT)): f.read_text() for f in files}
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, "paper/"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    keep = [n for n in names if n == "paper/main.tex" or
            (n.startswith(("paper/sections/", "paper/tables/")) and n.endswith(".tex"))]
    return {n: subprocess.run(["git", "show", f"{rev}:{n}"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout for n in keep}


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def numbers(text):
    text = strip_comments(text).replace("$-$", "-").replace("\\%", "%")
    return {m.group(0).replace("−", "-").lstrip("+-") for m in NUM.finditer(text)}


def json_numbers():
    vals = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            x = abs(float(o))
            for d in range(0, 5):          # the value as it would print at each precision
                vals.add(f"{x:.{d}f}")
                vals.add(f"{100 * x:.{d}f}")
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in [*glob.glob(str(ROOT / "outputs/v4/analysis/*.json")),
              str(ROOT / "paper/tables/numbers.json")]:
        if Path(f).exists():
            walk(json.load(open(f)))
    return vals


def norm(n):
    """Canonical string at the number's OWN printed precision ('.5' -> '0.5')."""
    return "0" + n if n.startswith(".") else n


def headings(texts):
    h = set()
    for name, s in texts.items():
        for m in re.finditer(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}", strip_comments(s)):
            h.add(f"{m.group(1)}: {m.group(2)}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="git ref (commit or tag) of the earlier paper")
    a = ap.parse_args()
    cur, base = tex_files(), tex_files(a.baseline)
    cur_nums = set().union(*(numbers(t) for t in cur.values()))
    base_nums = set().union(*(numbers(t) for t in base.values()))
    cur_n, base_n, js = {norm(n) for n in cur_nums}, {norm(n) for n in base_nums}, json_numbers()
    new = sorted(n for n in cur_n if n not in base_n)
    untraced = [n for n in new if n not in js]
    new_traced = [n for n in new if n in js]
    dropped = sorted(n for n in base_n if n not in cur_n)

    flags = []
    for name, s in cur.items():
        body = strip_comments(s)
        for pat in FORBIDDEN:
            for m in re.finditer(pat, body, flags=re.I):
                a0 = max(body.rfind(". ", 0, m.start()), body.rfind("\n\n", 0, m.start())) + 1
                e0 = body.find(". ", m.end())
                sent = body[a0: e0 if e0 > 0 else len(body)].replace("\n", " ").strip()
                if any(re.search(c, sent, flags=re.I) for c in ALLOW_CONTEXT):
                    continue
                flags.append(f"{name}: '{m.group(0)}' in: {sent[:220]}")

    hb, hc = headings(base), headings(cur)
    print(f"== UNTRACED numbers ({len(untraced)}): need a source in the analysis outputs ==")
    for n in untraced:
        where = [f for f, t in cur.items() if n in {norm(x) for x in numbers(t)}]
        print(f"  {n:>10}  in {', '.join(where)}")
    print(f"\n== NEW numbers with a candidate source ({len(new_traced)}): verify each against its source ==")
    for n in new_traced:
        where = [f for f, t in cur.items() if n in {norm(x) for x in numbers(t)}]
        print(f"  {n:>10}  in {', '.join(where)}")
    print(f"\n== DROPPED baseline numbers ({len(dropped)}): confirm each was moved, not lost ==")
    print("  " + ", ".join(dropped) if dropped else "  none")
    print(f"\n== WORDING flags ({len(flags)}): check against paper/CLAIMS.md ==")
    for f in flags:
        print("  " + f)
    print("\n== Headings removed ==\n  " + ("\n  ".join(sorted(hb - hc)) or "none"))
    print("\n== Headings added ==\n  " + ("\n  ".join(sorted(hc - hb)) or "none"))
    sys.exit(1 if (untraced or flags) else 0)


if __name__ == "__main__":
    main()
