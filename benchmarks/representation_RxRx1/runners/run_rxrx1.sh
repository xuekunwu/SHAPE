#!/usr/bin/env bash
# representation_RxRx1: the launcher of one stage of the seven-arm benchmark (frozen, dinov3, shape, react, llm,
# openphenom, cellprofiler on RxRx1 well mAP), on an ALREADY ALLOCATED cluster node (GPU for control / shape /
# score / finish) or on the VM (RXRX_PLATFORM=vm, below). Never source it.
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh prepare    # once (CPU): tool inputs + pinned DINOv3
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh assets     # once, on a LOGIN node: OpenPhenom snapshot
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh control    # once per prep (GPU): frozen, dinov3(_s1,_s2)
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh external   # once per prep (GPU + CPUs): external arms
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh shape      # GPU: one SHAPE-arm run into a new run dir
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh llm        # GPU: LLM-written features (amendment A5)
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh react      # GPU: the ReAct arm (react_arm.py, the shared runner)
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh score-arm <arm> [<run-dir>]   # GPU: one arm into the score
#                                         # cache, its headline printed (llm / shape / react: that arm's run dir)
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh score <run-dir>    # GPU: well mAP of every arm
#   bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh finish <run-dir>   # GPU: score + export tarball
# Environment:
#   RXRX_EXPECTED_BRANCH  the branch this checkout must be on (default main)
#   RXRX_EXPECTED_COMMIT  required, the 40-hex commit of this checkout (every stage refuses another one)
#   CELL_TYPE             HEPG2 (default; task 116 never used it), HUVEC or RPE (config.py PREPS); with
#                         RXRX_PREP_SET=full_v4 or full_v5 also U2OS (config.py FULL_V4_PREPS, FULL_V5_PREPS)
#   RXRX_PREP_SET         v3 (default: the task-116 preps), full_v4 or full_v5 (the full-set preps, config.py
#                         PREP_SETS); a full_v4 stage passes --prep-set full_v4 and names its cache directory, run
#                         directories and logs by the run label <cell type>_full_v4 where v3 uses <cell type> (full_v5
#                         alike: <cell type>_full_v5)
#   MAX_ROUNDS            shape stage: rounds at most (default 8)
#   NO_ENGINE             shape stage: 1 = the deterministic fallback without a reasoner (a smoke only)
#   RXRX_SEED             shape stage: the analyzer's seed of every SHAPE call (default 0; A18, repeats of the arm)
#   LLM_RUN, REACT_RUN    score / finish stages: the llm / react run directories scored with the SHAPE run (optional;
#                         refused when made with other code -- llm: its llm design; react: react_arm.py and the
#                         shared runner -- or another prep)
#   CONTROL_SEEDS         control stage: comma-separated dinov3 seeds (default 0,1,2; 0 = dinov3, k = dinov3_s<k>);
#                         none = frozen only
#   EXTERNAL_ARMS         external stage: openphenom,cellprofiler (default) or one of them (cellprofiler alone: no GPU)
#   CP_WORKERS            external stage: cellprofiler processes (default SLURM_CPUS_PER_TASK, else nproc)
#   OPENPHENOM_INPUT_SIZE external stage: 256 only (OpenPhenom's positional grid is fixed); another value is refused
#   OPENPHENOM_FIT        external stage: resize (default) or pad (the crop centred on a zero canvas)
#   EXTERNAL_VENV         external stage: the venv with the external arms' packages (default .../venvs/shape_external)
#   EXTERNAL_LIMIT        external stage: a pilot on the first N cells into <cell type>/external_pilot/<setting>/
#                         (one directory per arm + fit + N, never overwritten, never scored)
# Job time (--time, or the interactive allocation): llm, shape and react alike (A5 as amended: the llm arm has SHAPE's
# MAX_ROUNDS attempts and SHAPE's wall-clock rule, the job's deadline; it needs 16 CPUs + 1 GPU), 16:00:00.
# The reasoner (gpt-5.6-luna) authenticates ONLY through the Stanford AI Gateway key from api.sh; the key is never
# printed or written to disk. Outputs, logs and caches go under BENCH (below; the SHAPE results layout,
# SHAPE_RESULTS_ROOT overrides its root); nothing is written under the repo, /home/users or /scratch/users, and a batch
# job whose Slurm output is not under BENCH/logs is refused. No crop file is
# written: the tool reads the task-116 images.npy as its packed-array input (crop_array).
# The VM mode (contract amendment A8): RXRX_PLATFORM=vm runs a stage on a machine without Slurm.
# Every cluster path below is a variable with its default:
#   RXRX_MAIN (MAIN), RXRX_DATA_ROOT, SHAPE_RESULTS_ROOT, RXRX_DINOV3_REPO, RXRX_DINOV3_WEIGHTS, RXRX_PY_ENV (the conda
#   env; on the VM a venv), RXRX_CONDA_SH, EXTERNAL_VENV, RXRX_API_ENV (exports LLM_API_KEY / LLM_BASE_URL; on the VM
#   mode 600). VM mode only: RXRX_RUN_TAG replaces SLURM_JOB_ID (default <hostname -s>-<UTC>), RXRX_DEADLINE_UTC
#   (YYYY-MM-DDTHH:MM:SSZ) is the deadline (none without it), RXRX_RESUME_RUN=<run dir> resumes that unfinished
#   shape / llm / react run in place (run_rxrx1.py --resume). Skipped there: the Slurm allocation, the scontrol log
#   check, `module load` and conda; kept: the commit, branch and clean-tree checks, the import checks, the GPU checks
#   and every check of the stages themselves.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'NOT_READY: invoke with bash; do not source this script.'
    return 2
fi
set -eo pipefail
umask 027

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EXPECTED_BRANCH=${RXRX_EXPECTED_BRANCH:-main}                # the branch this evaluation runs from
PLATFORM=${RXRX_PLATFORM:-cluster}                                            # cluster, or vm (contract A8)
MAIN=${RXRX_MAIN:-/scratch/groups/your-group/your-user}                              # scripts, logs, outputs
DATA_ROOT=${RXRX_DATA_ROOT:-/scratch/groups/your-group/your-user/shape_benchmark/RxRx/agentic_ssl}  # the v3 preps (read-only)
# The SHAPE results layout (one outlet for every SHAPE benchmark): everything a stage writes is under BENCH.
RESULTS_ROOT="${SHAPE_RESULTS_ROOT:-/scratch/groups/your-group/your-user/shape_benchmark/results}"
BENCH="$RESULTS_ROOT/representation_RxRx1"
OUT="$BENCH/cache"           # reusable intermediates: assets, <run label>/{prep,private,control,external*}
LOGS="$BENCH/logs"           # the Slurm logs (#SBATCH --output/--error) and the stage logs
# a SHAPE-arm run: $BENCH/<run_id>, run_id = shape_<run label>_<commit8>_<UTC>_<job id> (run label = the cell type,
# full_v4 / full_v5: <cell type>_<prep set>); an LLM-arm run alike, with llm_ or react_ in place of shape_
DINOV3_REPO=${RXRX_DINOV3_REPO:-/scratch/groups/your-group/your-user/model/dinov3/dinov3_pinned_6876159a}
DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
WEIGHTS=/scratch/groups/your-group/your-user/model/dinov3/dinov3_backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
WEIGHTS=${RXRX_DINOV3_WEIGHTS:-$WEIGHTS}
WEIGHTS_SHA256=73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c
CONDA_SH=${RXRX_CONDA_SH:-/home/groups/your-group/your-user/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${RXRX_PY_ENV:-/home/groups/your-group/your-user/miniconda3/shape_bench}  # the Python environment (VM: a venv)
# the external arms' own packages (timm, transformers 4.x, cp_measure, centrosome, mahotas, pillow < 12) live in a
# venv layered on CONDA_ENV (--system-site-packages), so CONDA_ENV itself is never changed (README: External arms)
EXTERNAL_VENV=${EXTERNAL_VENV:-/home/groups/your-group/your-user/venvs/shape_external}
API_SH=${RXRX_API_ENV:-/home/users/your-user/api.sh}
GATEWAY_V1=https://llm-gateway.example.edu/v1
CELL_TYPE=${CELL_TYPE:-HEPG2}
PREP_SET=${RXRX_PREP_SET:-v3}
MAX_ROUNDS=${MAX_ROUNDS:-8}
RXRX_SEED=${RXRX_SEED:-0}

die() { printf 'NOT_READY: %s\n' "$*" >&2; exit 2; }
[[ "$RXRX_SEED" =~ ^[0-9]+$ ]] || die "RXRX_SEED must be a non-negative integer"

stage=${1:-}
case "$stage" in prepare|assets|control|external|shape|llm|react|score-arm|score|finish) ;;
    *) die "stage must be prepare|assets|control|external|shape|llm|react|score-arm|score|finish" ;; esac
[[ "$PLATFORM" == cluster || "$PLATFORM" == vm ]] || die "RXRX_PLATFORM must be cluster or vm"
if [[ "$PLATFORM" == cluster ]]; then
    # assets downloads, so it runs on a login node (compute nodes load every model offline); every other stage needs
    # a job
    [[ -n "${SLURM_JOB_ID:-}" || "$stage" == assets ]] || die "run inside an existing Slurm allocation"
    jobtag=${SLURM_JOB_ID:-login-$(hostname -s)-$$}
else
    # the VM: no Slurm; the run tag (a host label plus a UTC start stamp, fixed by the driver) replaces the job id
    [[ -z "${SLURM_JOB_ID:-}" ]] || die "RXRX_PLATFORM=vm inside a Slurm job (SLURM_JOB_ID=$SLURM_JOB_ID)"
    jobtag=${RXRX_RUN_TAG:-$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)}
    [[ "$jobtag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RXRX_RUN_TAG must be letters, digits, '.', '_' or '-'"
    utc_format='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
    [[ -z "${RXRX_DEADLINE_UTC:-}" || "$RXRX_DEADLINE_UTC" =~ $utc_format ]] \
        || die "RXRX_DEADLINE_UTC must be YYYY-MM-DDTHH:MM:SSZ (or unset: no deadline)"
    export RXRX_PLATFORM=vm RXRX_RUN_TAG="$jobtag"
    [[ -z "${RXRX_DEADLINE_UTC:-}" ]] || export RXRX_DEADLINE_UTC
fi
[[ "${RXRX_EXPECTED_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || die "set RXRX_EXPECTED_COMMIT to the 40-hex commit"
[[ "$PREP_SET" =~ ^(v3|full_v4|full_v5)$ ]] || die "RXRX_PREP_SET must be v3, full_v4 or full_v5"
if [[ "$PREP_SET" == v3 ]]; then
    [[ "$CELL_TYPE" =~ ^(HEPG2|HUVEC|RPE)$ ]] || die "CELL_TYPE must be HEPG2, HUVEC or RPE"
    LABEL=$CELL_TYPE                   # the run label: the cache directory, run directory and log name of the prep
    prep_flag=()
else
    [[ "$CELL_TYPE" =~ ^(HEPG2|HUVEC|RPE|U2OS)$ ]] || die "CELL_TYPE must be HEPG2, HUVEC, RPE or U2OS ($PREP_SET)"
    LABEL=${CELL_TYPE}_$PREP_SET
    prep_flag=(--prep-set "$PREP_SET")
fi
[[ "$MAX_ROUNDS" =~ ^[0-9]+$ ]] || die "MAX_ROUNDS must be an integer"
export RXRX_EXPECTED_COMMIT
# Slurm writes a batch job's output files itself: they go to the benchmark's logs directory too.
job=$( [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true)
if [[ "$job" == *"BatchFlag=1"* ]]; then
    for key in StdOut StdErr; do
        file=$(grep -o "$key=[^ ]*" <<<"$job" | head -1 | cut -d= -f2- || true)
        [[ "$file" == "$LOGS"/* ]] || die "the job's $key is $file; use #SBATCH --output/--error under $LOGS"
    done
fi

if [[ "$PLATFORM" == cluster ]]; then
    module load system git/2.45.1  # system git is 1.8.3.1
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    [[ -f "$CONDA_ENV/pyvenv.cfg" && -f "$CONDA_ENV/bin/activate" ]] || die "RXRX_PY_ENV=$CONDA_ENV is not a venv"
    # shellcheck disable=SC1091
    source "$CONDA_ENV/bin/activate"
fi
hash -r
[[ "$(command -v python)" == "$CONDA_ENV/bin/python" ]] || die "wrong python: $(command -v python)"

# The code that runs must be the commit's: HEAD, the branch, no tracked changes, no untracked files in the
# benchmark or in SHAPE itself.
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RXRX_EXPECTED_COMMIT" ]] || die "checkout $REPO is not at $RXRX_EXPECTED_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$EXPECTED_BRANCH" ]] || die "checkout is not on $EXPECTED_BRANCH"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || die "checkout has tracked changes"
[[ -z "$(git -C "$REPO" status --porcelain -- benchmarks/representation_RxRx1 shape)" ]] \
    || die "checkout has uncommitted or untracked files under benchmarks/representation_RxRx1 or shape"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Every cache and temporary file of every stage (the import checks below included) on group scratch, never in ~.
cache="$OUT/jobs/${jobtag}"
for dir in "$BENCH" "$cache"; do
    [[ "$dir" == "$MAIN"/* ]] || die "$dir is not under $MAIN"
done
mkdir -p "$cache/tmp"
export TMPDIR="$cache/tmp" XDG_CACHE_HOME="$cache/xdg" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/nv" \
       MPLCONFIGDIR="$cache/mpl" NUMBA_CACHE_DIR="$cache/numba"
shape_file=$(python -c 'import shape; print(shape.__file__)')
[[ "$shape_file" == "$REPO/"* ]] || die "import shape resolves to $shape_file, not inside $REPO"

# Every library the stages import, checked once up front.
missing=$(python - <<'PY'
import importlib
bad = []
for name in ("numpy", "pandas", "scipy", "skimage", "torch", "torchvision", "huggingface_hub",
             "anndata", "scanpy", "leidenalg", "pydantic"):
    try:
        importlib.import_module(name)
    except Exception as error:  # noqa: BLE001
        bad.append(f"{name} ({type(error).__name__}: {error})")
print("; ".join(bad))
PY
)
[[ -z "$missing" ]] || die "$CONDA_ENV cannot import: $missing (conda install -c conda-forge <package>)"

utc() { date -u +%Y%m%dT%H%M%SZ; }
mkdir -p "$LOGS"
log="$LOGS/${stage}_${LABEL}_${jobtag}_$(utc).log"
printf 'commit=%s job=%s host=%s stage=%s cell_type=%s shape=%s log=%s\n' "$RXRX_EXPECTED_COMMIT" "$jobtag" \
    "$(hostname)" "$stage" "$CELL_TYPE" "$shape_file" "$log"
[[ "$PREP_SET" == v3 ]] || printf 'prep_set=%s run_label=%s\n' "$PREP_SET" "$LABEL"

# logged <command...>: output to the terminal and the log; stop on failure. A log file that cannot be written (a
# Lustre client eviction, errno 108) does not stop tee from passing the output on, so it never breaks the stage.
logged() {
    local status
    set +e
    "$@" 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    set -e
    [[ "$status" == 0 ]] || die "'$*' failed (rc=$status); log: $log"
}
rxrx() {
    python "$REPO/benchmarks/representation_RxRx1/run_rxrx1.py" "$@" --cell-type "$CELL_TYPE" --output-root "$OUT" \
        "${prep_flag[@]}"
}
# new_run <stage>: RUN, the stage's run directory -- a new one, or on the VM RXRX_RESUME_RUN (an unfinished run of this
# stage, run label and commit, resumed in place: resume_flag=(--resume))
resume_flag=()
new_run() {
    if [[ "$PLATFORM" == vm && -n "${RXRX_RESUME_RUN:-}" ]]; then
        RUN=${RXRX_RESUME_RUN%/}
        [[ "$RUN" == "$BENCH/${1}_${LABEL}_${RXRX_EXPECTED_COMMIT:0:8}_"* && "$RUN" != */../* \
           && -f "$RUN/RUN_META.json" ]] \
            || die "RXRX_RESUME_RUN=$RXRX_RESUME_RUN is not a $1 run of $LABEL at" \
                   "${RXRX_EXPECTED_COMMIT:0:8} under $BENCH"
        resume_flag=(--resume)
        printf 'RESUME_RUN=%s\n' "$RUN"
        return 0
    fi
    RUN="$BENCH/${1}_${LABEL}_${RXRX_EXPECTED_COMMIT:0:8}_$(utc)_${jobtag}"
    [[ ! -e "$RUN" ]] || die "$RUN exists"
}
# load_api: the Stanford AI Gateway credentials (RXRX_API_ENV, default API_SH). On the VM the file must be mode 600 or
# 400 and only LLM_API_KEY and LLM_BASE_URL are taken from it (read in a subshell: nothing else it sets or exports
# reaches this shell or the stage); the key is never printed.
load_api() {
    if [[ "$PLATFORM" == vm ]]; then
        local mode
        mode=$( [[ -f "$API_SH" ]] && stat -c %a "$API_SH" || true)
        [[ "$mode" == 600 || "$mode" == 400 ]] || die "$API_SH is missing or not mode 600 / 400"
        LLM_API_KEY=$(unset LLM_API_KEY; source "$API_SH" >/dev/null 2>&1; printf '%s' "${LLM_API_KEY:-}")
        LLM_BASE_URL=$(unset LLM_BASE_URL; source "$API_SH" >/dev/null 2>&1; printf '%s' "${LLM_BASE_URL:-}")
        return 0
    fi
    source "$API_SH" >/dev/null 2>&1 || true
}

if [[ "$stage" == prepare ]]; then
    for required in "$DINOV3_REPO/hubconf.py" "$WEIGHTS"; do
        [[ -e "$required" ]] || die "missing $required"
    done
    [[ "$(git -C "$DINOV3_REPO" rev-parse HEAD)" == "$DINOV3_COMMIT" ]] || die "DINOv3 source is not at $DINOV3_COMMIT"
    [[ -z "$(git -C "$DINOV3_REPO" status --porcelain --untracked-files=no)" ]] || die "DINOv3 source has tracked changes"
    [[ "$(sha256sum "$WEIGHTS" | cut -d' ' -f1)" == "$WEIGHTS_SHA256" ]] || die "DINOv3 weights sha256 mismatch"
    logged rxrx prepare --data-root "$DATA_ROOT" --dinov3-repo "$DINOV3_REPO" --weights "$WEIGHTS"
    printf 'STAGE_COMPLETE prepare -> %s/{%s,assets}\n' "$OUT" "$LABEL"
    exit 0
fi

if [[ "$stage" == assets ]]; then
    # OpenPhenom (Hugging Face recursionpharma/OpenPhenom, pinned revision in external.py) onto group scratch, once;
    # the Hub cache of the download stays in this job's cache directory.
    [[ -f "$OUT/assets/ASSETS.json" ]] || die "run the prepare stage first (it builds $OUT/assets)"
    export HF_HOME="$cache/hf" HF_HUB_DISABLE_TELEMETRY=1
    logged rxrx assets
    printf 'STAGE_COMPLETE assets -> %s/assets/openphenom/ASSETS.json\n' "$OUT"
    exit 0
fi

# ------------------------------------------------------------------ GPU stages
[[ -f "$OUT/$LABEL/prep/PREPARE.json" && -f "$OUT/assets/ASSETS.json" ]] || die "run the prepare stage first"
EXTERNAL_ARMS=${EXTERNAL_ARMS:-openphenom,cellprofiler}
needs_gpu=1
[[ "$stage" == external && ",$EXTERNAL_ARMS," != *,openphenom,* ]] && needs_gpu=0
if [[ "$needs_gpu" == 1 ]]; then
    nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
    # On an exclusive-process GPU any leftover process (e.g. an earlier run) locks every new one out.
    busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    [[ -z "$busy" ]] || die "the GPU is in use by: $busy (stop those processes first)"
    python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else "NOT_READY: no CUDA GPU visible")'
fi

if [[ "$stage" == external ]]; then
    [[ "$EXTERNAL_ARMS" =~ ^(openphenom|cellprofiler)(,(openphenom|cellprofiler))?$ ]] \
        || die "EXTERNAL_ARMS must be openphenom, cellprofiler or openphenom,cellprofiler"
    CP_WORKERS=${CP_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
    OPENPHENOM_INPUT_SIZE=${OPENPHENOM_INPUT_SIZE:-256}
    OPENPHENOM_FIT=${OPENPHENOM_FIT:-resize}
    [[ "$CP_WORKERS" =~ ^[0-9]+$ ]] || die "CP_WORKERS must be an integer"
    [[ "$OPENPHENOM_INPUT_SIZE" == 256 ]] || die "OPENPHENOM_INPUT_SIZE must be 256 (its fixed positional grid)"
    [[ "$OPENPHENOM_FIT" =~ ^(resize|pad)$ ]] || die "OPENPHENOM_FIT must be resize or pad"
    [[ -z "${EXTERNAL_LIMIT:-}" || "$EXTERNAL_LIMIT" =~ ^[1-9][0-9]*$ ]] \
        || die "EXTERNAL_LIMIT must be a positive integer"
    if [[ ",$EXTERNAL_ARMS," == *,openphenom,* ]]; then
        [[ -f "$OUT/assets/openphenom/ASSETS.json" ]] || die "no OpenPhenom assets: run the assets stage on a login node"
    fi
    # the arms' own libraries come from EXTERNAL_VENV, a venv of CONDA_ENV's python with --system-site-packages
    [[ -x "$EXTERNAL_VENV/bin/python" ]] || die "no $EXTERNAL_VENV (README: External arms, one-time venv setup)"
    if [[ "$PLATFORM" == cluster ]]; then          # the VM: a venv of its own; the import check below gates it
        grep -q "^home = $CONDA_ENV/bin$" "$EXTERNAL_VENV/pyvenv.cfg" \
            || die "$EXTERNAL_VENV was not made from $CONDA_ENV/bin/python"
        grep -q "^include-system-site-packages = true$" "$EXTERNAL_VENV/pyvenv.cfg" \
            || die "$EXTERNAL_VENV lacks --system-site-packages"
    fi
    # shellcheck disable=SC1091
    source "$EXTERNAL_VENV/bin/activate"
    hash -r
    [[ "$(command -v python)" == "$EXTERNAL_VENV/bin/python" ]] || die "wrong python: $(command -v python)"
    # (openphenom: timm, transformers; cellprofiler: cp_measure and its compiled dependencies)
    missing=$(EXTERNAL_ARMS="$EXTERNAL_ARMS" python - <<'PY'
import importlib, os
need = {"openphenom": ("timm", "transformers"), "cellprofiler": ("cp_measure", "mahotas", "centrosome")}
bad = []
for arm in os.environ["EXTERNAL_ARMS"].split(","):
    for name in need[arm]:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            bad.append(f"{name} ({type(error).__name__}: {error})")
if "openphenom" in os.environ["EXTERNAL_ARMS"].split(","):
    try:  # the snapshot's config was written with transformers 4.46.1; 5.x is untested with its code
        import transformers
        if int(transformers.__version__.split(".")[0]) >= 5:
            bad.append(f"transformers {transformers.__version__} (openphenom needs transformers < 5)")
    except Exception:  # noqa: BLE001 -- reported above when it cannot be imported
        pass
print("; ".join(bad))
PY
)
    [[ -z "$missing" ]] || die "$EXTERNAL_VENV cannot import: $missing (README: External arms)"
    # every model loads from group scratch: no Hub access from a compute node
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$cache/hf"
    logged rxrx external --arms "$EXTERNAL_ARMS" --workers "$CP_WORKERS" --input-size "$OPENPHENOM_INPUT_SIZE" \
        --fit "$OPENPHENOM_FIT" ${EXTERNAL_LIMIT:+--limit "$EXTERNAL_LIMIT"}
    where="external/{$EXTERNAL_ARMS}"
    [[ -z "${EXTERNAL_LIMIT:-}" ]] || where="external_pilot/<arm>[_$OPENPHENOM_FIT]_n$EXTERNAL_LIMIT"
    printf 'STAGE_COMPLETE external -> %s/%s/%s/EMBED.json (log %s)\n' "$OUT" "$LABEL" "$where" "$log"
    exit 0
fi

if [[ "$stage" == control ]]; then
    [[ -z "${CONTROL_SEEDS:-}" || "$CONTROL_SEEDS" == none || "$CONTROL_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]] \
        || die "CONTROL_SEEDS must be none, 0 or 0,1,2"
    if [[ "${CONTROL_SEEDS:-}" == none ]]; then logged rxrx control --seeds ""
    else logged rxrx control ${CONTROL_SEEDS:+--seeds "$CONTROL_SEEDS"}; fi
    printf 'STAGE_COMPLETE control -> %s/%s/control/CONTROL.json (log %s)\n' "$OUT" "$LABEL" "$log"
    exit 0
fi

if [[ "$stage" == score-arm ]]; then
    # one arm scored into the per-arm score cache (score.py) as soon as it exists; finish then reads it
    arm=${2:-}
    [[ "$arm" =~ ^(frozen|dinov3|dinov3_s[0-9]+|openphenom|cellprofiler|llm|shape|react)$ ]] \
        || die "usage: score-arm <frozen|dinov3|dinov3_s<k>|openphenom|cellprofiler|llm|shape|react> [<run-dir>]"
    run_flag=()
    if [[ "$arm" =~ ^(llm|shape|react)$ ]]; then
        [[ -f "${3:-}/RUN_META.json" ]] || die "score-arm $arm needs its run directory (with RUN_META.json)"
        run_flag=(--run "$(realpath "$3")")
    fi
    logged rxrx score-arm --arm "$arm" "${run_flag[@]}"
    printf 'STAGE_COMPLETE score-arm %s -> %s/%s/score_cache (log %s)\n' "$arm" "$OUT" "$LABEL" "$log"
    exit 0
fi

if [[ "$stage" == shape ]]; then
    new_run shape
    engine_flag=()
    if [[ "${NO_ENGINE:-0}" == 1 ]]; then
        engine_flag=(--no-engine)
        printf 'NO_ENGINE=1: the deterministic fallback, a smoke only\n'
    else
        # Stanford AI Gateway only: api.sh exports LLM_API_KEY with LLM_BASE_URL; the key is never printed. The base
        # URL must be the Stanford endpoint or nothing is sent; no other credential stays in the environment.
        load_api
        [[ -n "${LLM_API_KEY:-}" ]] || die "$API_SH did not export LLM_API_KEY"
        [[ "${LLM_BASE_URL%/}" == "$GATEWAY_V1" ]] || die "LLM_BASE_URL is '${LLM_BASE_URL:-unset}', expected $GATEWAY_V1"
        unset OPENAI_API_KEY OPENAI_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN LLM_MODEL ZHIPUAI_API_KEY \
              STANFORD_AI_API_ALIAS STANFORD_AI_USAGE_URL
        export LLM_API_KEY LLM_BASE_URL SHAPE_REQUIRE_AGENTIC=1
    fi
    printf 'RUN=%s\n' "$RUN"
    logged rxrx shape --run "$RUN" --max-rounds "$MAX_ROUNDS" --seed "$RXRX_SEED" "${engine_flag[@]}" \
        "${resume_flag[@]}"
    printf 'STAGE_COMPLETE shape -> %s (score it: bash %s score %s)\n' "$RUN" "$0" "$RUN"
    exit 0
fi

if [[ "$stage" == llm || "$stage" == react ]]; then
    new_run "$stage"
    # Stanford AI Gateway only (the shape stage's block): api.sh exports LLM_API_KEY with LLM_BASE_URL; the key is
    # never printed. The base URL must be the Stanford endpoint or nothing is sent; no other credential stays in the
    # environment. These arms have no deterministic fallback: without an engine the stage stops.
    load_api
    [[ -n "${LLM_API_KEY:-}" ]] || die "$API_SH did not export LLM_API_KEY"
    [[ "${LLM_BASE_URL%/}" == "$GATEWAY_V1" ]] || die "LLM_BASE_URL is '${LLM_BASE_URL:-unset}', expected $GATEWAY_V1"
    unset OPENAI_API_KEY OPENAI_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN LLM_MODEL ZHIPUAI_API_KEY \
          STANFORD_AI_API_ALIAS STANFORD_AI_USAGE_URL
    export LLM_API_KEY LLM_BASE_URL
    printf 'RUN=%s\n' "$RUN"
    budget_flag=()                      # llm: its attempts are SHAPE's rounds (A5 as amended 2026-10-13)
    [[ "$stage" != llm ]] || budget_flag=(--max-rounds "$MAX_ROUNDS")
    logged rxrx "$stage" --run "$RUN" "${budget_flag[@]}" "${resume_flag[@]}"
    # COMPLETE (written last) only for a run that delivered; a failed run (no valid reply, a failed call) has none
    if python -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "delivered")' \
            "$RUN/submission/decision.json"; then
        date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/COMPLETE"
    else
        printf 'the %s run did not deliver (see %s/submission/decision.json); no COMPLETE written\n' "$stage" "$RUN"
    fi
    printf 'STAGE_COMPLETE %s -> %s (score it with the SHAPE run: %s_RUN=%s bash %s score <shape-run-dir>)\n' \
        "$stage" "$RUN" "${stage^^}" "$RUN" "$0"
    exit 0
fi

[[ -d "${2:-}" ]] || die "usage: $stage <run-dir>"
RUN=$(realpath "$2")
[[ -f "$RUN/RUN_META.json" ]] || die "$RUN is not a run directory (no RUN_META.json)"
# the LLM comparison arms (amendments A4, A5), scored with this run when given
arm_runs=()
arm_dirs=()
if [[ -n "${LLM_RUN:-}" ]]; then
    [[ -f "$LLM_RUN/RUN_META.json" ]] || die "LLM_RUN=$LLM_RUN is not a run directory (no RUN_META.json)"
    arm_runs+=(--llm-run "$(realpath "$LLM_RUN")")
    arm_dirs+=("$(realpath "$LLM_RUN")")
fi
if [[ -n "${REACT_RUN:-}" ]]; then
    [[ -f "$REACT_RUN/RUN_META.json" ]] || die "REACT_RUN=$REACT_RUN is not a run directory (no RUN_META.json)"
    arm_runs+=(--react-run "$(realpath "$REACT_RUN")")
    arm_dirs+=("$(realpath "$REACT_RUN")")
fi
logged rxrx "$stage" --run "$RUN" "${arm_runs[@]}"
if [[ "$stage" == finish ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN/COMPLETE"      # written last: the run scored and exported
    if [[ "$PLATFORM" == vm ]]; then
        for dir in "$RUN" "${arm_dirs[@]}"; do printf 'EXPORT %s/EXPORT_%s.tar.gz\n' "$dir" "$(basename "$dir")"; done
    else
        printf 'Download on your local machine:\n  scp "your-user@your-cluster:%s/EXPORT_*.tar.gz" .\n' "$RUN"
        for dir in "${arm_dirs[@]}"; do                 # each LLM-arm run's own export (finish writes it there)
            printf '  scp "your-user@your-cluster:%s/EXPORT_*.tar.gz" .\n' "$dir"
        done
    fi
fi
printf 'STAGE_COMPLETE %s -> %s/RESULT.md (log %s)\n' "$stage" "$RUN" "$log"
