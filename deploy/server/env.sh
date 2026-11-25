# Shared settings for the server bundle v6 scripts (sourced, not run).
# Override any of these on the command line, e.g.  WORKERS=8 bash 5_run_experiments.sh
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ZIP="${REPO_ZIP:-$( (ls -1 "$BUNDLE_DIR"/FISM2_flare_repro_repo_v*.zip 2>/dev/null || true) | sort -V | tail -1)}"   # empty if the zip was removed
REPO="${REPO:-$BUNDLE_DIR/fism2_repro}"            # unpacked code + data
VENV="${VENV:-$BUNDLE_DIR/.venv}"                  # python virtual environment
BASE_PY="${BASE_PY:-python3}"                      # python used to create the venv (>= 3.10)
LOGDIR="$BUNDLE_DIR/run_logs"
mkdir -p "$LOGDIR"
if [ "${USE_CURRENT_ENV:-0}" = "1" ]; then           # e.g. an activated conda env
  PY="${PY:-$(command -v python || command -v python3)}"
elif [ -x "$VENV/bin/python" ]; then
  PY="${PY:-$VENV/bin/python}"
else
  PY="${PY:-$BASE_PY}"
fi
# parallel training processes. On GPU each process uses < 1 GB of GPU memory; 8 total workers spread across GPUs is ideal.
# nproc obeys OMP_NUM_THREADS, which this file exports as 1 further down: when a step script is started from
# run_everything.sh (env.sh already sourced once) a plain `nproc` returns 1 and every experiment runs with one
# worker (the v6 clean run of 2026-11-10 took 10 h for this reason). Count the cores without those variables.
_NC=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 4); _W=$(( _NC / 2 > 0 ? _NC / 2 : 1 )); [ "$_W" -gt 8 ] && _W=8
WORKERS="${WORKERS:-$_W}"
# training device: auto = a healthy GPU with >= 2 GB free and < 90 % utilisation (workers spread over the GPUs
# with > 4 GB free; free memory / utilisation read from nvidia-smi). Override: FISM_DEVICE=cuda:1
export FISM_DEVICE="${FISM_DEVICE:-auto}"
# never train on the CPU silently: if no GPU is usable, wait up to FISM_GPU_WAIT_S for one, then stop with an
# error (a per-job CUDA out-of-memory still retries on another GPU, then on the CPU). FISM_REQUIRE_GPU=0 allows CPU.
export FISM_REQUIRE_GPU="${FISM_REQUIRE_GPU:-1}" FISM_GPU_WAIT_S="${FISM_GPU_WAIT_S:-900}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Auto-detect and skip broken GPUs (e.g. GPU 2 with '[Unknown Error]') + honour optional SKIP_GPUS=2:
# Sets CUDA_VISIBLE_DEVICES to the UUIDs (GPU-...) of all healthy GPUs so CUDA index shifting never drops a healthy GPU.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  _SMI="$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -n "$_SMI" ]; then
    _TOTAL=$(echo "$_SMI" | awk -F', *' '$2 ~ /^GPU-/ {n++} END {print n+0}')
    _VIS=$(echo "$_SMI" | awk -F', *' -v skip=",${SKIP_GPUS:-}," '
      $2 ~ /^GPU-/ && index(skip, ","$1",") == 0 && $3 ~ /^[0-9.]+$/ && $4 ~ /^[0-9.]+$/ {print $2}
    ' | paste -sd,)
    _GOOD=$(echo "$_VIS" | awk -F',' 'NF>0 {print NF; exit} END {if (NR==0) print 0}')
    if [ -n "$_VIS" ] && { [ -n "${SKIP_GPUS:-}" ] || [ "$_GOOD" -lt "$_TOTAL" ]; }; then
      export CUDA_VISIBLE_DEVICES="$_VIS"
      echo "[env] Auto-selected healthy GPU UUIDs ($_GOOD/$_TOTAL): CUDA_VISIBLE_DEVICES=$_VIS"
    fi
  fi
fi
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES
# worker processes are forked from the parent; FISM_MP_CONTEXT=spawn starts fresh interpreters instead
[ -n "${FISM_MP_CONTEXT:-}" ] && export FISM_MP_CONTEXT
# one BLAS/OpenMP thread per process, so WORKERS really bounds the CPU use
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
DRY_RUN="${DRY_RUN:-0}"

log()  { echo "[$(date '+%F %T')] $*"; }
die()  { echo "[$(date '+%F %T')] ERROR: $*" >&2; exit 1; }
run()  { log "\$ $*"; if [ "$DRY_RUN" = "1" ]; then return 0; fi; "$@"; }
need_repo() { [ -d "$REPO/scripts/download" ] || die "code not unpacked yet - run: bash 1_setup.sh"; }

# latest dataset built by 4_download_all.sh (events_latis.npz, events_latis_v2.npz, ...)
latest_events() {
  ls -1 "$REPO"/data/processed/events_latis*.npz 2>/dev/null | sort -V | tail -1
}
