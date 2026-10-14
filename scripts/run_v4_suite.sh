#!/usr/bin/env bash
# EXPERIMENT SUITE. Runs every GPU job the paper needs, then the CPU analysis.
# Decision criteria are pre-registered in the paper (Appendix B).
#
#   E1  word-position vs answer-position probe      scripts/run_wordpos.py      (stage: wordpos)
#   E2  retained fraction above the nuisance floor  analysis of E1 outputs      (stage: analyze)
#   E3  ceiling-aware representation-readout gap    analysis of grid outputs    (stage: analyze)
#   E4  estimator fixes on the main grid            scripts/run_grid.py         (stage: grid)
#   E5  question x scale interaction test           analysis of E1 outputs      (stage: analyze)
#   E6  length-matched count tasks (set "lm")       included in grid + wordpos
#   X   cross-task control, fixed estimator         scripts/run_crosstask.py    (stages: crosstask,
#                                                                                 crosstask_term)
# Every GPU job uses ONE estimator: held-out-word GroupKFold (3 folds), 12 evenly spaced layers,
# held-out layer selection, cold-start folds, sorted-set sampler. Outputs go to NEW directories
# (outputs/v4/*, outputs/bigstim_v4_grid/); nothing published is overwritten.
#
# USAGE (on the GPU box, from the repo root)
#   bash scripts/run_v4_suite.sh                         # smoke, then every stage
#   bash scripts/run_v4_suite.sh --stages wordpos,analyze   # E1 first: the go/no-go experiment
#   bash scripts/run_v4_suite.sh --dry-run               # print the job list only
#   bash scripts/run_v4_suite.sh --grid-shots 0,1,3,5    # also refresh the ICL appendix shots
#
# USAGE (from the laptop, via .env.deploy: REMOTE_HOST, SSH_PORT, SSH_KEY, REMOTE_DIR, HF_TOKEN)
#   bash scripts/run_v4_suite.sh --remote [user@host] [stage flags...]   # sync + launch in tmux
#   bash scripts/run_v4_suite.sh --pull                                  # fetch outputs + logs
#   bash scripts/run_v4_suite.sh --analyze-only                          # CPU analysis, locally
#
# STAGES (default order; --stages takes any comma-separated subset, always run in this order)
#   smoke           Qwen3-0.6B, n=60, every script once          ~10 min   aborts suite on failure
#   wordpos         16 models x shots {0,3} x sets orig,cvs,lm   ~16-22 GPU-h
#   grid            16 models x shots {0,3} x sets orig,cvs,lm   ~14-20 GPU-h  (gate+align+drop,
#                   rank control N=3; free generation at shots=3)
#   crosstask       16 models, shots 0, sets orig,cvs            ~10-16 GPU-h
#   crosstask_term  16 models, term-label secondary arm          ~4-6 GPU-h
#   analyze         CPU; writes outputs/v4/analysis/summary{,_peak}.{json,md}
#   stemorder       F3 follow-up (opt-in: name it in --stages); 16 models, shots 0   ~4-6 GPU-h
#   review          reviewer R1-R3 (opt-in): token split + MLP probe + same-draw continuation ~5-8 GPU-h
# Total ~45-65 GPU-h on one 48 GB A6000. Every job passes --skip-existing, so an interrupted run
# resumes by re-running the same command. A failing job is logged and the suite continues.

set -uo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

ALL_STAGES="smoke,wordpos,grid,crosstask,crosstask_term,analyze"
STAGES="$ALL_STAGES"
MODELS_FILE="scripts/models.txt"
DEVICE="cuda"
GRID_SHOTS="0,3"
WORDPOS_SHOTS="0,3"
PROBE_LAYERS=12
N_BOOT=1000
N_RAND=3
N=800
WORDS="src/stimulus_words.txt"
ENV_FILE="$PROJECT_ROOT/.env.deploy"
DRY_RUN=false; REMOTE=false; PULL=false; HOST_OVERRIDE=""
PASS_ARGS=()

usage() { sed -n '2,40p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)         STAGES="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --models-file)    MODELS_FILE="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --grid-shots)     GRID_SHOTS="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --wordpos-shots)  WORDPOS_SHOTS="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --probe-layers)   PROBE_LAYERS="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --n-boot)         N_BOOT="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --device)         DEVICE="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --no-smoke)       STAGES="${STAGES//smoke,/}"; PASS_ARGS+=("$1"); shift ;;
        --analyze-only)   STAGES="analyze"; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --remote)         REMOTE=true
                          if [[ $# -gt 1 && "$2" == *@* ]]; then HOST_OVERRIDE="$2"; shift; fi
                          shift ;;
        --pull)           PULL=true; shift ;;
        --env-file)       ENV_FILE="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- remote mode
if $REMOTE || $PULL; then
    [ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }
    REMOTE_HOST="${HOST_OVERRIDE:-${REMOTE_HOST:-}}"
    [ -n "$REMOTE_HOST" ] && [ "$REMOTE_HOST" != "ubuntu@gpu.example.com" ] || {
        echo "ERROR: set REMOTE_HOST in $ENV_FILE or pass --remote user@host"; exit 1; }
    SSH_OPTS="-p ${SSH_PORT:-22} ${SSH_EXTRA_OPTS:-}"; [ -n "${SSH_KEY:-}" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    RDIR="${REMOTE_DIR:-~/char-store-probing}"
    if $PULL; then
        mkdir -p outputs logs
        rsync -avz --exclude analysis/ -e "ssh $SSH_OPTS" "$REMOTE_HOST:$RDIR/outputs/v4/" outputs/v4/
        rsync -avz -e "ssh $SSH_OPTS" "$REMOTE_HOST:$RDIR/outputs/bigstim_v4_grid/" outputs/bigstim_v4_grid/
        rsync -avz -e "ssh $SSH_OPTS" "$REMOTE_HOST:$RDIR/logs/v4/" logs/v4/ || true
        echo "Pulled. Run: bash scripts/run_v4_suite.sh --analyze-only"
        exit 0
    fi
    echo "--- syncing code to $REMOTE_HOST:$RDIR ---"
    ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p $RDIR"
    rsync -avz --exclude .venv --exclude __pycache__ --exclude outputs --exclude checkpoints \
        --exclude logs --exclude data --exclude .git --exclude wandb --exclude '*.pyc' \
        -e "ssh $SSH_OPTS" "$PROJECT_ROOT/" "$REMOTE_HOST:$RDIR/"
    [ -f "$ENV_FILE" ] && rsync -az -e "ssh $SSH_OPTS" "$ENV_FILE" "$REMOTE_HOST:$RDIR/.env.deploy"
    ARGSTR="$(printf '%q ' "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}")"
    # UV_UNMANAGED_INSTALL: no receipt in ~/.config/uv (root-owned on some images) and no PATH edits
    ssh $SSH_OPTS "$REMOTE_HOST" bash -s <<REMOTE || { echo "ERROR: remote setup/launch failed (see output above)"; exit 1; }
set -eo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd $RDIR
ABS_DIR="\$(pwd)"   # REMOTE_DIR may be relative to ~; tmux must not re-cd it
# tmux runs the command via sh -c, which strips printf %q escapes (no extra quoting layer)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="\$HOME/.local/bin" sh
uv sync
mkdir -p logs/v4
tmux kill-session -t charprobe 2>/dev/null || true
tmux new-session -d -s charprobe -c "\$ABS_DIR" \
    "cd \"\$ABS_DIR\" && bash scripts/run_v4_suite.sh $ARGSTR 2>&1 | tee -a \"\$ABS_DIR/logs/v4/launch.log\""
sleep 5
tmux has-session -t charprobe 2>/dev/null || {
    echo "ERROR: tmux session charprobe exited immediately. Last lines of logs/v4/launch.log:"
    tail -30 logs/v4/launch.log; exit 1; }
echo "launched tmux session charprobe in \$ABS_DIR"
REMOTE
    echo "Attach: ssh $SSH_OPTS $REMOTE_HOST -t tmux attach -t charprobe"
    echo "Pull:   bash scripts/run_v4_suite.sh --pull"
    exit 0
fi

# --------------------------------------------------------------------------- local run
[ -f "$ENV_FILE" ] && { set -a; source "$ENV_FILE"; set +a; }     # HF_TOKEN for gated models
unset ORTHO_LEGACY_SET_ORDER                                     # v4 = sorted-set sampler
export PATH="$HOME/.local/bin:$PATH"

read_models() {
    while IFS= read -r line; do
        line="${line%%#*}"; line="$(echo "$line" | xargs)"; [ -n "$line" ] && echo "$line"
    done < "$1"
}
MODELS=(); while IFS= read -r m; do MODELS+=("$m"); done < <(read_models "$MODELS_FILE")
[ "${#MODELS[@]}" -gt 0 ] || { echo "ERROR: no models in $MODELS_FILE"; exit 1; }

bs_for() {   # forward batch size per model (48 GB); word-position runs keep 3 arms in memory
    case "$1" in
        *13B*|*/phi-4) echo 16 ;;
        *-8B*|*-9b*|*-7B*|*Hybrid*|*OLMoE*) echo 32 ;;
        *) echo 48 ;;
    esac
}
extra_for() { case "$1" in *Phi-4-mini*) echo "--no-remote-code" ;; *) echo "" ;; esac; }
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

LOG_DIR="logs/v4"; mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/suite_$RUN_ID.log"
FAILED=()
N_OK=0

run() {   # $1 = job name, rest = command
    local name="$1"; shift
    if $DRY_RUN; then echo "[$name] $*"; return 0; fi
    local log="$LOG_DIR/${name}.log"
    echo "$(date '+%F %T') START $name" | tee -a "$MASTER_LOG"
    if "$@" >>"$log" 2>&1; then
        N_OK=$((N_OK+1)); echo "$(date '+%F %T') OK    $name" | tee -a "$MASTER_LOG"
    else
        FAILED+=("$name"); echo "$(date '+%F %T') FAIL  $name  (see $log)" | tee -a "$MASTER_LOG"
        return 1
    fi
}

tag() { basename "$1" | tr '.-' '__'; }
COMMON="--device $DEVICE --words-file $WORDS --probe-layers $PROBE_LAYERS --skip-existing"

if ! $DRY_RUN && [[ "$STAGES" != "analyze" ]]; then
    uv sync >/dev/null
    uv run python -c "import torch; assert torch.cuda.is_available() or '$DEVICE' != 'cuda', 'no CUDA'; print('device OK:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '$DEVICE')" \
        || { echo "ERROR: device check failed"; exit 1; }
fi
echo "=== v4 suite: stages=$STAGES models=${#MODELS[@]} grid_shots=$GRID_SHOTS wordpos_shots=$WORDPOS_SHOTS ===" | tee -a "$MASTER_LOG"

# --- smoke: every script once on the smallest model, tiny n, separate output dirs -------------
if has_stage smoke; then
    SM="Qwen/Qwen3-0.6B"
    ok=true
    run smoke_wordpos uv run python scripts/run_wordpos.py --device "$DEVICE" --model $SM --shots 0 \
        --sets orig,lm --n 60 --batch-size 16 --probe-layers 4 --bootstrap 20 \
        --out-dir outputs/v4_smoke/wordpos || ok=false
    run smoke_grid uv run python scripts/run_grid.py --device "$DEVICE" --model $SM --shots 3 --sets lm \
        --n 60 --batch-size 16 --probe-layers 4 --layer-select heldout --bootstrap 20 --cold-start \
        --rank-control 2 --gen-shots 3 --out-tag v4_smoke_grid || ok=false
    run smoke_crosstask uv run python scripts/run_crosstask.py --device "$DEVICE" --model $SM --shots 0 \
        --sets cvs --n 60 --batch-size 16 --probe-layers 4 --b-tasks word_length,most_frequent_letter \
        --cold-start --layer-select heldout --out-dir outputs/v4_smoke/crosstask || ok=false
    run smoke_analyze uv run python scripts/analyze_v4.py --smoke --perms 50 || ok=false
    if ! $ok && ! $DRY_RUN; then echo "SMOKE FAILED -- aborting before any full run." | tee -a "$MASTER_LOG"; exit 1; fi
fi

# --- E1 (+E2/E5 inputs, v4 question-deletion control) ---------------------------------------
if has_stage wordpos; then
    for m in "${MODELS[@]}"; do
        run "wordpos_$(tag "$m")" uv run python scripts/run_wordpos.py $COMMON --model "$m" \
            --shots "$WORDPOS_SHOTS" --sets orig,cvs,lm --n $N --batch-size "$(bs_for "$m")" \
            --bootstrap $N_BOOT --out-dir outputs/v4/wordpos $(extra_for "$m")
    done
fi

# --- F3 stem-order follow-up: question before vs after the word (shots=0) --------------------
if has_stage stemorder; then
    for m in "${MODELS[@]}"; do
        run "stemorder_$(tag "$m")" uv run python scripts/run_wordpos.py $COMMON --model "$m" \
            --stem-order --shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" \
            --bootstrap $N_BOOT --out-dir outputs/v4/stemorder $(extra_for "$m")
    done
fi

# --- reviewer follow-ups R1-R3 (opt-in: name "review" in --stages) ---------------------------
if has_stage review; then
    for m in "${MODELS[@]}"; do
        run "review_$(tag "$m")" uv run python scripts/run_review.py --device "$DEVICE" \
            --words-file $WORDS --probe-layers $PROBE_LAYERS --skip-existing --model "$m" \
            --shots 0,3 --mlp-shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" \
            --out-dir outputs/v4/review $(extra_for "$m")
        run "contarm_$(tag "$m")" uv run python scripts/run_crosstask.py $COMMON --model "$m" \
            --shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" --only-cont-arm \
            --cold-start --layer-select heldout --out-dir outputs/v4/crosstask $(extra_for "$m")
        run "contarm_term_$(tag "$m")" uv run python scripts/run_crosstask.py $COMMON --model "$m" \
            --shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" --only-cont-arm \
            --include-term-tasks --a-tasks extraction,letter_before --out-tag term \
            --cold-start --layer-select heldout --out-dir outputs/v4/crosstask $(extra_for "$m")
    done
fi

# --- E4 main grid (+E3, E6): fixed estimator, rank control, free generation -----------------
if has_stage grid; then
    for m in "${MODELS[@]}"; do
        run "grid_$(tag "$m")" uv run python scripts/run_grid.py $COMMON --model "$m" \
            --shots "$GRID_SHOTS" --sets orig,cvs,lm --n $N --batch-size "$(bs_for "$m")" \
            --layer-select heldout --bootstrap $N_BOOT --cold-start --rank-control $N_RAND \
            --gen-shots 3 --out-tag v4_grid $(extra_for "$m")
    done
fi

# --- cross-task control, fixed estimator ----------------------------------------------------
if has_stage crosstask; then
    for m in "${MODELS[@]}"; do
        run "crosstask_$(tag "$m")" uv run python scripts/run_crosstask.py $COMMON --model "$m" \
            --shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" \
            --cold-start --layer-select heldout --out-dir outputs/v4/crosstask $(extra_for "$m")
    done
fi
if has_stage crosstask_term; then
    for m in "${MODELS[@]}"; do
        run "crosstask_term_$(tag "$m")" uv run python scripts/run_crosstask.py $COMMON --model "$m" \
            --shots 0 --sets orig,cvs --n $N --batch-size "$(bs_for "$m")" \
            --include-term-tasks --a-tasks extraction,letter_before --out-tag term \
            --cold-start --layer-select heldout --out-dir outputs/v4/crosstask $(extra_for "$m")
    done
fi

# --- analysis (CPU) ------------------------------------------------------------------------
if has_stage analyze; then
    run analyze_hsel uv run python scripts/analyze_v4.py --estimator hsel
    run analyze_peak uv run python scripts/analyze_v4.py --estimator peak
fi

if $DRY_RUN; then exit 0; fi
echo "=== done: $N_OK ok, ${#FAILED[@]} failed ===" | tee -a "$MASTER_LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '  failed: %s\n' "${FAILED[@]}" | tee -a "$MASTER_LOG"
    echo "Re-run the same command to retry (completed cells are skipped)." | tee -a "$MASTER_LOG"
    exit 1
fi
