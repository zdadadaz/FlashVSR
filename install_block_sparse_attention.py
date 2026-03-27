#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Block-Sparse-Attention Installation Helper for FlashVSR
=========================================================
Detects GPU/CUDA compatibility and installs Block-Sparse-Attention
from official repo or pre-compiled wheels.

Usage:
    python install_block_sparse_attention.py [--force] [--wheel-only]

Options:
    --force      Force reinstall even if already installed
    --wheel-only Skip compilation, only use pre-compiled wheel
    --check      Only check installation status
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import platform
import torch


def get_cuda_version():
    """Get CUDA version from PyTorch."""
    if not torch.cuda.is_available():
        return None, None
    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = cuda_version.split(".")
        return int(major), int(minor)
    return None, None


def get_torch_version():
    """Get PyTorch version string."""
    return torch.__version__


def get_gpu_name():
    """Get GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU detected"


def get_python_version():
    """Get Python version."""
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def check_cuda_compatibility():
    """Check if CUDA version supports Block-Sparse-Attention."""
    major, minor = get_cuda_version()
    if major is None:
        return False, "CUDA not available"
    if major < 12 or (major == 12 and minor < 8):
        return False, f"CUDA {major}.{minor} — Block-Sparse-Attention requires CUDA >= 12.8"
    return True, f"CUDA {major}.{minor} ✓"


def check_bsa_installed():
    """Check if Block-Sparse-Attention is already installed."""
    try:
        import block_sparse_attention
        ver = getattr(block_sparse_attention, "__version__", "unknown")
        # Try to import the function
        from block_sparse_attention import block_sparse_attention_func
        return True, ver
    except ImportError:
        return False, None


def get_prebuilt_wheel_url(cuda_version, torch_version, python_version):
    """Get pre-built wheel URL from smthemex fork or Quark."""
    # smthemex provides wheels for cu128 + torch2.8 + py311
    base = "https://github.com/smthemex/Block-Sparse-Attention/releases/download/wheels"
    
    # Map CUDA to compute capability suffix
    # cu128 = CUDA 12.8
    if cuda_version == "12.8":
        plat = "win_amd64" if platform.system() == "Windows" else "linux_x86_64"
        fname = f"block_sparse_attention-1.0.0-{plat}.whl"
        return f"{base}/{fname}"
    return None


def install_from_official():
    """Install Block-Sparse-Attention from official MIT-HAN-Lab repo."""
    print("[1/3] Cloning official Block-Sparse-Attention repo...")
    repo_dir = "/tmp/Block-Sparse-Attention"
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir])
    
    subprocess.run(
        ["git", "clone", "https://github.com/mit-han-lab/Block-Sparse-Attention.git", repo_dir],
        check=True
    )
    
    print("[2/3] Installing build dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "packaging", "ninja"], check=True)
    
    print("[3/3] Building Block-Sparse-Attention (this may take 5-10 minutes)...")
    os.chdir(repo_dir)
    result = subprocess.run(
        [sys.executable, "setup.py", "install"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Build failed:\n{result.stderr[-2000:]}")
        return False
    return True


def install_from_smethemex_fork():
    """Install from smthemex fork with pre-built wheels."""
    print("[1/2] Cloning smthemex Block-Sparse-Attention fork...")
    repo_dir = "/tmp/Block-Sparse-Attention-smthemex"
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir])
    
    subprocess.run(
        ["git", "clone", "https://github.com/smthemex/Block-Sparse-Attention.git", repo_dir],
        check=True
    )
    subprocess.run(["git", "checkout", "main"], cwd=repo_dir)
    
    print("[2/2] Running pip install...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-build-isolation", "."],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Install failed:\n{result.stderr[-2000:]}")
        return False
    return True


def install_from_quark_wheel():
    """Download and install pre-built wheel from Quark (requires curl/wget)."""
    cuda_ver, torch_ver, py_ver = "cu128", "torch2.8", "py311"
    plat = "win_amd64" if platform.system() == "Windows" else "linux_x86_64"
    wheel_name = f"block_sparse_attention-1.0.0-{plat}.whl"
    quark_url = f"https://pan.quark.cn/s/c9ba067c89bc"
    
    print(f"[1/2] Pre-built wheel: {wheel_name}")
    print(f"[2/2] Download from: {quark_url}")
    print()
    print("⚠️  Manual download required for Quark link.")
    print(f"   Download {wheel_name} and run:")
    print(f"   pip install /path/to/{wheel_name}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Block-Sparse-Attention Installer for FlashVSR")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--wheel-only", action="store_true", help="Skip compilation, use wheel")
    parser.add_argument("--check", action="store_true", help="Only check installation status")
    args = parser.parse_args()

    print("=" * 60)
    print("Block-Sparse-Attention Installation Helper")
    print("=" * 60)
    print()
    
    # System info
    print(f"GPU:         {get_gpu_name()}")
    cuda_ok, cuda_msg = check_cuda_compatibility()
    print(f"CUDA:        {cuda_msg}")
    print(f"PyTorch:    {get_torch_version()}")
    print(f"Python:     {get_python_version()}")
    print(f"Platform:  {platform.system()}")
    print()
    
    # Check existing installation
    installed, version = check_bsa_installed()
    if installed:
        print(f"✅ Block-Sparse-Attention already installed (version: {version})")
        if args.force:
            print("Reinstalling as requested...")
        else:
            print("Use --force to reinstall.")
            return
    
    if args.check:
        print("❌ Block-Sparse-Attention NOT installed")
        return
    
    # Installation recommendations
    print("=" * 60)
    if not cuda_ok:
        print("⚠️  CUDA version incompatible with Block-Sparse-Attention.")
        print("   Install from smthemex fork (pre-built) instead:")
        print("   python install_block_sparse_attention.py --wheel-only")
        print()
        print("   Or use sparse_sage_attention (no compilation needed):")
        print("   sparse_sage_attention is already included and used by default")
        return
    
    major, minor = get_cuda_version()
    print(f"✅ CUDA {major}.{minor} is compatible!")
    print()
    
    # Strategy selection
    if args.wheel_only:
        print("Installing from smthemex fork (pre-built)...")
        if install_from_smethemex_fork():
            print("✅ Installation successful!")
    else:
        # Try smthemex first (usually works), fall back to official
        print("Strategy: smthemex fork → official repo")
        print()
        
        print("Step 1: Try smthemex fork (pre-built wheels)...")
        if install_from_smethemex_fork():
            print("✅ Installation successful!")
            return
        
        print()
        print("Step 2: Build from official repo (this may take 10+ minutes)...")
        if install_from_official():
            print("✅ Installation successful!")
            return
        
        print()
        print("❌ All installation methods failed.")
        print("   Fallback: sparse_sage_attention works without Block-Sparse-Attention.")
        print("   FlashVSR will run correctly but with slightly lower quality.")
    
    # Verify
    print()
    installed, version = check_bsa_installed()
    if installed:
        print(f"✅ Verified: Block-Sparse-Attention {version} installed")
    else:
        print("⚠️  Could not verify installation")


if __name__ == "__main__":
    main()
