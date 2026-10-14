#!/usr/bin/env python3
"""Unembedding-alignment analysis: WHY does compute-readout alignment never form?

The gate showed the char answer is present at the FINAL layer (probe ~1.0 for letters,
~0.8 for counts) yet the output head reads it only for retrieval, only at scale. This asks
whether that is a subspace-alignment property of the output head.

At the final layer (post final-norm) the model can only distinguish answer classes along
the answer-token unembedding directions U_ans = W_U[answer_token_ids].  We:
  1. probe the answer class from the full final-layer rep H            -> acc_full  (present?)
  2. project H onto span(centered U_ans)  =  the ONLY directions the output head uses to
     tell answer tokens apart, then re-probe                          -> acc_proj  (readable?)
  3. readable_fraction = (acc_proj - chance) / (acc_full - chance)
  4. subspace_overlap  = captured energy of the class-mean subspace inside span(U_ans)

Prediction: retrieval (last_letter) readable_fraction rises to ~1 with scale; counting stays
low at all scales -> the count is encoded ORTHOGONAL to the output head's answer-token
readout subspace (a null-space / basis-misalignment that scale never fixes -> explains why
no minimal intervention routed it).

Usage:  uv run python scripts/run_alignment.py --model <hf-id> --shots 3 [--no-remote-code]
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import run_stage1`
from src.config import ExperimentConfig, apply_env_overrides
from src.model import CausalLM, load_tokenizer
from src.intervention import final_norm, lm_head
import run_stage1 as s1  # probe_acc_gpu (random split) + probe_acc_grouped (held-out-word)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def orthobasis(rows, tol=1e-6):
    """Orthonormal basis (d, r) of the row space of a MEAN-CENTERED (m, d) matrix."""
    M = rows - rows.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = int((S > tol * (S[0] + 1e-12)).sum())
    return Vt[:r].T  # (d, r)


def random_subspace(d, r, seed):
    """Orthonormal basis (d, r) of a UNIFORMLY random r-dim subspace of R^d (QR of a Gaussian)."""
    G = np.random.RandomState(seed).randn(d, r)
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


def pca_subspace(H, r):
    """Orthonormal basis (d, r) of the top-r principal directions of H (mean-centred, LABEL-FREE,
    so it is a fair rank-matched comparator for the answer-token subspace)."""
    M = H - H.mean(axis=0, keepdims=True)
    # right singular vectors of the centred data = principal directions
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, Vt.shape[0])
    return Vt[:r].T


def alignment_metrics(model, tokenizer, data, task_list, device, group_probe, ex_words, folds=5,
                      rank_control=0, rank_seed=0):
    """Unembedding-alignment gap per task from an already-extracted `data` (must include the final
    layer). Shared by main() and the unified run_grid.py so the forward pass is computed once.
    group_probe -> held-out-word (leakage-safe) probe. Returns {task: {...}}.

    rank_control > 0 adds the RANK-MATCHED controls the review demands (§3.3): acc_proj <= acc_full
    holds by construction (nested models), so the projected accuracy only means something against
    other rank-r_out projections. We add
      acc_rand : mean acc over `rank_control` random r_out-dim subspaces (+ sd, + min/max)
      acc_pca  : acc on the top-r_out principal components of H (label-free)
    Costs (rank_control + 1) extra probe fits per task, NO extra forward passes. Default 0 = off,
    so the published numbers are reproduced exactly."""
    tasks, targets = data["tasks"], data["target_ids"]
    n_layers = max(data["ans_h"])              # final layer = hidden_states[n_layers]
    Hf = data["ans_h"][n_layers]
    norm, head = final_norm(model), lm_head(model)
    hd = next(head.parameters()).dtype
    with torch.no_grad():
        Hn = norm(torch.tensor(Hf, device=device).to(hd)).float().cpu().numpy()
    W_U = head.weight.detach().float().cpu().numpy()
    ex_words = np.asarray(ex_words)
    results = {}
    for t in task_list:
        m = tasks == t
        if m.sum() == 0:
            continue
        H, y, gwords = Hn[m], targets[m], ex_words[m]
        sp = sorted(set(s1.answer_space_ids(tokenizer, t)))
        classes, counts = np.unique(y, return_counts=True)
        keep = [int(c) for c, n in zip(classes, counts) if n >= 5 and int(c) in sp]
        if len(keep) < 2:
            continue
        Q_out = orthobasis(W_U[np.array(keep)])                 # output-head answer-token subspace
        mu = np.stack([H[y == c].mean(axis=0) for c in keep])
        Q_rep = orthobasis(mu)
        r_rep = Q_rep.shape[1]
        overlap = float(((Q_out.T @ Q_rep) ** 2).sum() / max(r_rep, 1)) if r_rep else 0.0
        Hp = (H @ Q_out) @ Q_out.T
        if group_probe:
            full = s1.probe_acc_grouped(H, y, gwords, device, folds=folds)
            proj = s1.probe_acc_grouped(Hp, y, gwords, device, folds=folds)
        else:
            full = s1.probe_acc_gpu(H, y, device, folds=3)
            proj = s1.probe_acc_gpu(Hp, y, device, folds=3)
        if not full or not proj:
            continue
        chance = full["chance"]
        readable = (proj["acc"] - chance) / max(full["acc"] - chance, 1e-6)
        r_out = int(Q_out.shape[1])
        entry = {
            "n_classes": full["n_classes"], "chance": round(chance, 3),
            "majority": round(full.get("majority", chance), 4),
            "acc_full": round(full["acc"], 4), "acc_proj": round(proj["acc"], 4),
            "readable_fraction": round(float(np.clip(readable, 0, 1.5)), 3),
            "subspace_overlap": round(overlap, 4), "grouped": bool(group_probe),
            "dims": {"r_rep": r_rep, "r_out": r_out},
        }
        if rank_control > 0:
            def _probe(X):
                r = (s1.probe_acc_grouped(X, y, gwords, device, folds=folds) if group_probe
                     else s1.probe_acc_gpu(X, y, device, folds=3))
                return r["acc"] if r else None

            rand_accs = []
            for i in range(rank_control):
                Qr = random_subspace(H.shape[1], r_out, rank_seed + 1000 * i + len(t))
                a = _probe((H @ Qr) @ Qr.T)
                if a is not None:
                    rand_accs.append(a)
            a_pca = None
            Qp = pca_subspace(H, r_out)
            if Qp.shape[1] == r_out:
                a_pca = _probe((H @ Qp) @ Qp.T)
            entry["rank_matched"] = {
                "r_out": r_out, "n_random": len(rand_accs),
                "acc_rand_mean": round(float(np.mean(rand_accs)), 4) if rand_accs else None,
                "acc_rand_sd": round(float(np.std(rand_accs)), 4) if rand_accs else None,
                "acc_rand_all": [round(a, 4) for a in rand_accs],
                "acc_pca": round(a_pca, 4) if a_pca is not None else None,
                # the quantities the paper should report instead of a bare acc_full-acc_proj:
                "proj_minus_rand": (round(proj["acc"] - float(np.mean(rand_accs)), 4)
                                    if rand_accs else None),
                "proj_minus_pca": round(proj["acc"] - a_pca, 4) if a_pca is not None else None,
            }
        results[t] = entry
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--out", default="outputs/stage1_align/align.json")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--set", dest="taskset", default="orig", choices=list(s1.TASK_SETS),
                    help="task set: 'orig' or 'cvs' (copy-vs-synthesize battery)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="forward batch size (lower for big models at high shots to avoid OOM)")
    ap.add_argument("--group-probe", action="store_true",
                    help="leakage-safe: probe acc_full/acc_proj with GroupKFold BY QUERY WORD "
                         "(no word in train+test). Writes to outputs/robustness/. Robustness check "
                         "that readable_fraction is not driven by word->answer memorization.")
    ap.add_argument("--folds", type=int, default=5, help="held-out-word folds when --group-probe")
    ap.add_argument("--rank-control", type=int, default=0, metavar="N",
                    help="CONTROL: also probe N random r_out-dim subspaces + the top-r_out PCs, so "
                         "acc_proj can be read against RANK-MATCHED alternatives. "
                         "0 = off (published behaviour). Adds no forward passes.")
    ap.add_argument("--words-file", default=None,
                    help="large stimulus word list (src/stimulus_words.txt) -> leakage-safe pool; "
                         "outputs go to bigstim_robustness/. Omit for the legacy ~100-word pool.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip if the output file already exists (cheap restarts)")
    args = ap.parse_args()

    task_list = s1.TASK_SETS[args.taskset]
    config = ExperimentConfig()
    mt = f"_{s1.model_tag(args.model)}" if args.model else ""
    if args.model:
        config.model.model_name = args.model
    if args.group_probe:
        rob = "bigstim_robustness" if args.words_file else "robustness"
        args.out = f"outputs/{rob}/aligngrouped_{args.taskset}{mt}_s{args.shots}.json"
    else:
        if args.taskset != "orig":
            args.out = args.out.replace("stage1_align/", f"stage1_align_{args.taskset}/")
        if args.words_file:
            args.out = args.out.replace("outputs/stage1_align", "outputs/bigstim_stage1_align")
        args.out = args.out.replace(".json", f"{mt}_s{args.shots}.json")
    if args.skip_existing and Path(args.out).exists():
        logger.info("skip (exists): %s", args.out); return
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False

    logger.info("Loading %s on %s", config.model.model_name, args.device)
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    words = s1.load_word_pool(args.words_file)
    logger.info("word pool: %d words (%s)", len(words), args.words_file or "built-in ~100-word pool")
    examples = s1.build_examples(words, args.n, args.seed, shots=args.shots,
                                 tokenizer=tokenizer, tasks=task_list)
    ex_words = np.array([e["word"] for e in examples])  # for held-out-word probe (--group-probe)

    n_layers = model.base_model.config.num_hidden_layers  # hidden_states[n_layers] = final layer
    data = s1.extract(model, tokenizer, examples, args.device,
                      batch_size=args.batch_size, keep_layers=[n_layers], want_word=False)
    results = alignment_metrics(model, tokenizer, data, task_list, args.device,
                                args.group_probe, ex_words, folds=args.folds,
                                rank_control=args.rank_control, rank_seed=args.seed)
    for t in task_list:
        if t in results:
            r = results[t]
            logger.info("%-16s acc_full=%.2f acc_proj=%.2f readable=%.2f overlap=%.2f", t,
                        r["acc_full"], r["acc_proj"], r["readable_fraction"], r["subspace_overlap"])

    out = {"gap": results, "meta": {"model": config.model.model_name, "shots": args.shots,
                                    "n_layers": n_layers, "seed": args.seed, "set": args.taskset,
                                    "grouped": bool(args.group_probe), "words_file": args.words_file,
                                    "n_words": len(words)}}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("UNEMBEDDING ALIGNMENT   (present at final layer? readable via answer tokens?)")
    print("=" * 78)
    for t in task_list:
        if t not in results:
            continue
        r = results[t]
        print(f"{t:16s} acc_full={r['acc_full']:.2f}  acc_proj={r['acc_proj']:.2f}  "
              f"readable={r['readable_fraction']:.2f}  overlap={r['subspace_overlap']:.2f}")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
