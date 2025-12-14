#!/usr/bin/env bash
set -euo pipefail

# Install multiple venvs targeting CUDA 12 (container nvcc shows CUDA 12.4)
# - Creates: .perceval-venv, .qiskit-venv, .pennylane-venv, .cuquantum-venv, .cirq-venv
# - Installs cuquantum-cu12 and cuquantum-python-cu12 into .cuquantum-venv
#
# Usage:
#   chmod +x install_venvs_cuda12_cuquantum.sh
#   ./install_venvs_cuda12_cuquantum.sh
#
# Run this from the repository root (e.g. /workspace/MathsHPC-Labs-Docker).

CUDA_VERSION="12.4.0"
REPO_ROOT="$(pwd)"
echo "Repo root: $REPO_ROOT"
echo "Target CUDA version: $CUDA_VERSION"

PERCEVAL_VENV="$REPO_ROOT/.perceval-venv"
QISKIT_VENV="$REPO_ROOT/.qiskit-venv"
PENNYLANE_VENV="$REPO_ROOT/.pennylane-venv"
CUQUANTUM_VENV="$REPO_ROOT/.cuquantum-venv"
CIRQ_VENV="$REPO_ROOT/.cirq-venv"

# Ensure basic system prerequisites (attempt to install if running as root)
need_sudo=false
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found in container. Install python3 first." >&2
  exit 1
fi

# Install build deps if apt is present and running as root (useful for compiling wheels)
if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
  echo "Installing common build dependencies (python3-venv, pip, build-essential)..."
  apt-get update -y
  apt-get install -y --no-install-recommends python3-venv python3-pip build-essential python3-dev
fi

# helper to create venv and pip install packages
create_and_install() {
  local venv_dir="$1"; shift
  local pkgs=( "$@" )
  if [ -d "$venv_dir" ]; then
    echo "venv exists: $venv_dir"
  else
    python3 -m venv "$venv_dir"
    echo "created: $venv_dir"
  fi
  "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
  echo "Installing into $venv_dir: ${pkgs[*]}"
  if ! "$venv_dir/bin/pip" install --upgrade "${pkgs[@]}"; then
    echo "WARNING: pip install failed for: ${pkgs[*]}" >&2
  fi

  # quick import checks
  for m in "${pkgs[@]}"; do
    mod="$(echo "$m" | cut -d'=' -f1)"
    # map cupy-cuda12x -> cupy, cuquantum-python-cu12 -> cuquantum or cuquantum_python
    case "$mod" in
      cupy-cuda*) mod="cupy";;
      cuquantum-python-*) mod="cuquantum";;
      cuquantum-*) mod="cuquantum";;
    esac
    if "$venv_dir/bin/python" -c "import importlib, sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)"; then
      echo " - import $mod: OK"
    else
      echo " - import $mod: FAILED (may require extra deps or different wheel)"
    fi
  done
}

echo
echo "Checking CUDA/toolkit visibility inside container:"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not found (CUDA toolkit may be missing in container)."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader || true
fi

echo
echo "1) Perceval"
create_and_install "$PERCEVAL_VENV" "perceval"

echo
echo "2) Qiskit"
create_and_install "$QISKIT_VENV" "qiskit"

echo
echo "3) PennyLane (and pennylane-lightning CPU backend)"
create_and_install "$PENNYLANE_VENV" "pennylane" "pennylane-lightning"

echo
echo "4) cuQuantum environment (cupy-cuda12x + cuquantum-cu12 + cuquantum-python-cu12)"
# Install cupy wheel for CUDA 12.x then NVIDIA cuQuantum wheels specified by NVIDIA
create_and_install "$CUQUANTUM_VENV" "cupy-cuda12x"
echo "Installing cuQuantum wheels recommended by NVIDIA..."
if ! "$CUQUANTUM_VENV/bin/pip" install --upgrade cuquantum-cu12 cuquantum-python-cu12; then
  echo "WARNING: cuquantum pip install failed. You may need to obtain a matching wheel from NVIDIA or ensure container's CUDA toolkit matches." >&2
fi

# quick manual guidance if import fails will be printed below

echo
echo "5) Cirq"
create_and_install "$CIRQ_VENV" "cirq"

# Write env file to quickly source venv locations and helpers
ENV_SH="$REPO_ROOT/env_venvs_cuda12.sh"
cat > "$ENV_SH" <<EOF
# env_venvs_cuda12.sh - venv locations and activation helpers (CUDA 12 target)
export REPO_ROOT="$REPO_ROOT"
export CUDA_VERSION="$CUDA_VERSION"
export PERCEVAL_VENV="$PERCEVAL_VENV"
export QISKIT_VENV="$QISKIT_VENV"
export PENNYLANE_VENV="$PENNYLANE_VENV"
export CUQUANTUM_VENV="$CUQUANTUM_VENV"
export CIRQ_VENV="$CIRQ_VENV"

activate-perceval() { [ -f "\$PERCEVAL_VENV/bin/activate" ] && source "\$PERCEVAL_VENV/bin/activate" || echo "No venv at \$PERCEVAL_VENV"; }
activate-qiskit()   { [ -f "\$QISKIT_VENV/bin/activate" ] && source "\$QISKIT_VENV/bin/activate" || echo "No venv at \$QISKIT_VENV"; }
activate-pennylane(){ [ -f "\$PENNYLANE_VENV/bin/activate" ] && source "\$PENNYLANE_VENV/bin/activate" || echo "No venv at \$PENNYLANE_VENV"; }
activate-cuquantum(){ [ -f "\$CUQUANTUM_VENV/bin/activate" ] && source "\$CUQUANTUM_VENV/bin/activate" || echo "No venv at \$CUQUANTUM_VENV"; }
activate-cirq()     { [ -f "\$CIRQ_VENV/bin/activate" ] && source "\$CIRQ_VENV/bin/activate" || echo "No venv at \$CIRQ_VENV"; }
EOF
chmod 644 "$ENV_SH"
echo "Wrote $ENV_SH"

# System-wide or per-user sourcing
if [ "$(id -u)" -eq 0 ] && [ -d /etc/profile.d ]; then
  cp "$ENV_SH" /etc/profile.d/mh_venvs_cuda12.sh
  chmod 644 /etc/profile.d/mh_venvs_cuda12.sh
  echo "Installed /etc/profile.d/mh_venvs_cuda12.sh for new sessions."
else
  if ! grep -Fq "source \"$ENV_SH\"" "$HOME/.bashrc" 2>/dev/null; then
    echo "source \"$ENV_SH\"" >> "$HOME/.bashrc"
    echo "Appended 'source $ENV_SH' to $HOME/.bashrc (effective in new shells)."
  fi
fi

# Add venvs to .gitignore to avoid accidental commits
GITIGNORE="$REPO_ROOT/.gitignore"
touch "$GITIGNORE"
for d in ".perceval-venv" ".qiskit-venv" ".pennylane-venv" ".cuquantum-venv" ".cirq-venv"; do
  entry="/$d/"
  grep -Fxq "$entry" "$GITIGNORE" || echo "$entry" >> "$GITIGNORE"
done
echo "Added venv entries to $GITIGNORE (if not already present)."

echo
echo "=== Installation finished ==="
echo "To use the venvs now:"
echo "  source $ENV_SH"
echo "  activate-cuquantum    # activate cuquantum venv"
echo "  python -c 'import cupy; print(cupy.__version__)'  # quick check inside activated venv"
echo
echo "If cuquantum import fails after installation:"
echo " - Ensure container has CUDA 12 toolkit (nvcc --version should show 12.4)."
echo " - If pip install cuquantum-cu12 / cuquantum-python-cu12 failed or import fails, download the wheel matching CUDA 12.4 from NVIDIA and install manually:"
echo "     source $CUQUANTUM_VENV/bin/activate"
echo "     pip install /path/to/cuquantum-<version>-cuda12*.whl"
echo
echo "Note: MX250 has only 2 GiB GPU RAM — many GPU workloads will OOM. Installation of packages is fine but runtime may be limited."