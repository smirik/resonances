#!/bin/bash
set -euo pipefail

# =============================================================================
# Deploy resonances simulation to a remote server
#
# Usage:
#   ./scripts/deploy.sh --ip 79.112.108.70 --port 47847 --dir /workspace
#   ./scripts/deploy.sh --ip 79.112.108.70 --port 47847 --dir /workspace \
#       --branch v1.0.0 --upload-astdys \
#       --sim-folder experiments/lkr-ias15 \
#       --sim-file experiments/lkr-ias15/sim_1_99999.py --background
# =============================================================================

# --- Defaults ---
SSH_USER="root"
SSH_KEY=""
GIT_BRANCH="main"
UPLOAD_ASTDYS=false
SIM_FOLDER=""
SIM_FILE=""
TELEGRAM_URL=""
BACKGROUND=false
DRY_RUN=false
REPO_URL="https://github.com/smirik/resonances.git"

# --- Required (set to empty) ---
IP=""
PORT=""
REMOTE_DIR=""

usage() {
    cat <<EOF
Usage: $0 --ip IP --port PORT --dir REMOTE_DIR [options]

Required:
  --ip IP              Server IP address
  --port PORT          SSH port
  --dir DIR            Remote working directory (e.g., /workspace)

Optional:
  --user USER          SSH user (default: root)
  --key KEYFILE        SSH identity file
  --branch BRANCH      Git branch (default: main)
  --upload-astdys      Upload local cache/allnum.cat to server
  --sim-folder FOLDER  Local folder to upload via rsync (e.g., experiments/lkr-ias15)
  --sim-file FILE      Simulation file to run on server (relative to repo root)
  --telegram-url URL   Telegram bot URL for completion notifications
  --background         Run simulation with nohup in background
  --dry-run            Print commands without executing
  -h, --help           Show this help
EOF
    exit 0
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --ip)        IP="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --dir)       REMOTE_DIR="$2"; shift 2 ;;
        --user)      SSH_USER="$2"; shift 2 ;;
        --key)       SSH_KEY="$2"; shift 2 ;;
        --branch)    GIT_BRANCH="$2"; shift 2 ;;
        --upload-astdys) UPLOAD_ASTDYS=true; shift ;;
        --sim-folder) SIM_FOLDER="$2"; shift 2 ;;
        --sim-file)  SIM_FILE="$2"; shift 2 ;;
        --telegram-url) TELEGRAM_URL="$2"; shift 2 ;;
        --background) BACKGROUND=true; shift ;;
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Validate required ---
if [[ -z "$IP" || -z "$PORT" || -z "$REMOTE_DIR" ]]; then
    echo "Error: --ip, --port, and --dir are required."
    usage
fi

# --- Build SSH/rsync options ---
SSH_OPTS="-p $PORT"
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi
SSH_CMD="ssh $SSH_OPTS $SSH_USER@$IP"
RSYNC_SSH="ssh $SSH_OPTS"
RSYNC_DEST="$SSH_USER@$IP:$REMOTE_DIR/resonances"

# --- Helpers ---
remote() {
    local cmd="$1"
    if $DRY_RUN; then
        echo "[dry-run] $SSH_CMD <<< $cmd"
    else
        echo ">>> remote: ${cmd:0:80}..."
        $SSH_CMD bash -s <<< "$cmd"
    fi
}

run_local() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

# =============================================================================
echo "=== Deploy resonances to $SSH_USER@$IP:$PORT ==="
echo "    Remote dir: $REMOTE_DIR"
echo "    Branch:     $GIT_BRANCH"
$DRY_RUN && echo "    *** DRY RUN ***"
echo ""

# --- Step 1: Install system packages + uv ---
echo "--- Step 1: System setup ---"
SUDO=""
if [[ "$SSH_USER" != "root" ]]; then
    SUDO="sudo"
fi
remote "which git > /dev/null 2>&1 && which curl > /dev/null 2>&1 && which gcc > /dev/null 2>&1 && echo 'system packages already installed' || ($SUDO apt-get update -qq && $SUDO apt-get install -y -qq git curl build-essential python3-dev > /dev/null 2>&1 && echo 'system packages installed')"

remote "which uv > /dev/null 2>&1 && echo 'uv already installed' || (curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -1)"

# --- Step 2: Clone or update repo ---
echo ""
echo "--- Step 2: Clone/update repo ---"
remote "cd $REMOTE_DIR && if [ -d resonances/.git ]; then
    cd resonances && git fetch --all -q && git checkout $GIT_BRANCH -q && git pull -q && echo 'repo updated to $GIT_BRANCH'
else
    git clone -b $GIT_BRANCH $REPO_URL && echo 'repo cloned ($GIT_BRANCH)'
fi"

# --- Step 3: Install Python dependencies ---
echo ""
echo "--- Step 3: Install dependencies ---"
remote "export PATH=\$HOME/.local/bin:\$PATH && cd $REMOTE_DIR/resonances && uv sync 2>&1 | tail -1"

# --- Step 4: Smoke test ---
echo ""
echo "--- Step 4: Smoke test ---"
remote 'export PATH=$HOME/.local/bin:$PATH && cd '"$REMOTE_DIR"'/resonances && uv run python -c "import resonances; print('"'"'resonances OK'"'"')" && uv run python -c "import rebound; print(f'"'"'rebound {rebound.__version__}'"'"')" && echo "CPUs: $(nproc)"'

# --- Step 5: Upload astdys catalog ---
if $UPLOAD_ASTDYS; then
    echo ""
    echo "--- Step 5: Upload AstDyS catalog ---"
    if [[ -f cache/allnum.cat ]]; then
        remote "mkdir -p $REMOTE_DIR/resonances/cache"
        run_local rsync -avz --progress -e "$RSYNC_SSH" cache/allnum.cat "$RSYNC_DEST/cache/"
    elif [[ -f cache/allnum.csv ]]; then
        remote "mkdir -p $REMOTE_DIR/resonances/cache"
        run_local rsync -avz --progress -e "$RSYNC_SSH" cache/allnum.csv "$RSYNC_DEST/cache/"
    else
        echo "WARNING: No cache/allnum.cat or cache/allnum.csv found locally. Skipping."
    fi
fi

# --- Step 6: Upload simulation folder ---
if [[ -n "$SIM_FOLDER" ]]; then
    echo ""
    echo "--- Step 6: Upload simulation folder ---"
    if [[ -d "$SIM_FOLDER" ]]; then
        # Ensure parent directory exists on server
        local_parent=$(dirname "$SIM_FOLDER")
        remote "mkdir -p $REMOTE_DIR/resonances/$local_parent"
        run_local rsync -avz --progress -e "$RSYNC_SSH" "$SIM_FOLDER" "$RSYNC_DEST/$local_parent/"
    else
        echo "ERROR: Local folder '$SIM_FOLDER' not found."
        exit 1
    fi
fi

# --- Step 7: Run simulation ---
if [[ -n "$SIM_FILE" ]]; then
    echo ""
    echo "--- Step 7: Run simulation ---"
    LOGFILE="$REMOTE_DIR/resonances/$(basename "$SIM_FILE" .py).log"

    # Build env prefix with optional Telegram URL
    ENV_PREFIX="export PATH=\$HOME/.local/bin:\$PATH"
    if [[ -n "$TELEGRAM_URL" ]]; then
        ENV_PREFIX="$ENV_PREFIX && export TELEGRAM_NOTIFY_URL='$TELEGRAM_URL'"
    fi

    if $BACKGROUND; then
        echo "Starting in background. Log: $LOGFILE"
        remote "$ENV_PREFIX && cd $REMOTE_DIR/resonances && nohup uv run python $SIM_FILE > $LOGFILE 2>&1 &"
        echo ""
        echo "Monitor with:"
        echo "  ssh $SSH_OPTS $SSH_USER@$IP \"tail -f $LOGFILE\""
    else
        echo "Running (blocking)..."
        remote "$ENV_PREFIX && cd $REMOTE_DIR/resonances && uv run python $SIM_FILE"
    fi
fi

echo ""
echo "=== Done ==="
