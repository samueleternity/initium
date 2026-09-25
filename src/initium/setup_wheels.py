"""Install the CUDA-matched binary wheels that PyPI metadata cannot express.

Run once after installing initium:

    initium-setup                # or: python -m initium.setup_wheels
    initium-setup --dry-run      # print the commands only

What it does (stdlib only, no torch reinstall):
  1. pytorch-dnc from git (PyPI's `dnc` is stale; direct URLs are forbidden in PyPI metadata).
  2. mamba-ssm + causal-conv1d prebuilt wheels matching the ACTIVE torch/CUDA build, from
     the wheel index, so nothing is compiled locally.
Steps that are already satisfied are skipped, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata

WHEEL_INDEX_TEMPLATE = "https://wheels.astral.sh/simple/{cuda_tag}/"
MAMBA_SSM_VERSION = "2.3.2.post1"
CAUSAL_CONV1D_VERSION = "1.6.2.post1"
DNC_SOURCE = "git+https://github.com/ixaxaar/pytorch-dnc.git"  # pin "@<sha>" for reproducibility
SKIPPABLE = ("dnc", "gpu-wheels")


@dataclass(frozen=True)
class TorchEnv:
    version: str  # "2.10.0"
    major_minor: str  # "2.10"
    cuda: str | None  # "12.8", or None for CPU-only torch

    @property
    def cuda_tag(self) -> str | None:
        return None if self.cuda is None else "cu" + self.cuda.replace(".", "")


def detect_torch() -> TorchEnv:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to install GPU wheels. Install a compatible PyTorch build first."
        ) from exc
    version = torch.__version__.split("+", 1)[0]
    return TorchEnv(version, ".".join(version.split(".")[:2]), torch.version.cuda)


def gpu_requirements(env: TorchEnv) -> list[str]:
    local = f"+cu.{env.cuda}.torch.{env.major_minor}"
    return [
        f"mamba-ssm=={MAMBA_SSM_VERSION}{local}",
        f"causal-conv1d=={CAUSAL_CONV1D_VERSION}{local}",
    ]


def resolve_installer(choice: str) -> str:
    if choice != "auto":
        return choice
    if importlib.util.find_spec("pip") is not None:
        return "pip"
    if shutil.which("uv") is not None:
        return "uv"
    raise SystemExit("Neither pip nor uv is available in this environment.")


def _base_cmd(installer: str) -> list[str]:
    if installer == "pip":
        return [sys.executable, "-m", "pip", "install"]
    return ["uv", "pip", "install", "--python", sys.executable]


def dnc_command(installer: str) -> list[str]:
    force = ["--force-reinstall"] if installer == "pip" else ["--reinstall-package", "dnc"]
    return [*_base_cmd(installer), *force, "--no-deps", DNC_SOURCE]


def gpu_command(installer: str, requirements: list[str], index_url: str) -> list[str]:
    cmd = _base_cmd(installer)
    if installer == "uv":
        # uv's default "first-index" strategy would stop at PyPI (which lacks the +cu local
        # versions) and never look at the extra index.
        cmd += ["--index-strategy", "unsafe-best-match"]
    return [*cmd, *requirements, "--extra-index-url", index_url]


def _installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _dnc_from_git() -> bool:
    try:
        raw = metadata.distribution("dnc").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False
    return bool(raw) and "vcs_info" in json.loads(raw)


def _gpu_satisfied(requirements: list[str]) -> bool:
    for req in requirements:
        name, version = req.split("==", 1)
        if _installed_version(name) != version:
            return False
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="initium-setup",
        description="Install CUDA-matched mamba-ssm / causal-conv1d wheels and pytorch-dnc (git).",
    )
    p.add_argument("--installer", choices=("auto", "pip", "uv"), default="auto")
    p.add_argument(
        "--skip",
        nargs="+",
        choices=SKIPPABLE,
        default=[],
        metavar="STEP",
        help=f"steps to skip: {', '.join(SKIPPABLE)}",
    )
    p.add_argument(
        "--wheel-index",
        default=WHEEL_INDEX_TEMPLATE,
        help="wheel index URL; may contain the {cuda_tag} placeholder (e.g. cu128)",
    )
    p.add_argument("--force", action="store_true", help="reinstall even if already satisfied")
    p.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    installer = resolve_installer(args.installer)

    plan: list[tuple[str, list[str]]] = []
    if "dnc" not in args.skip and (args.force or not _dnc_from_git()):
        plan.append(("pytorch-dnc (git)", dnc_command(installer)))
    if "gpu-wheels" not in args.skip:
        env = detect_torch()
        print(f"[initium-setup] torch={env.version} cuda={env.cuda or 'none (CPU build)'}")
        if env.cuda_tag is None:
            print("[initium-setup] CPU-only torch: skipping mamba-ssm/causal-conv1d (need CUDA).")
        else:
            reqs = gpu_requirements(env)
            if args.force or not _gpu_satisfied(reqs):
                index = args.wheel_index.format(cuda_tag=env.cuda_tag)
                plan.append(("GPU wheels", gpu_command(installer, reqs, index)))

    if not plan:
        print("[initium-setup] nothing to do - everything is already installed.")
        return 0
    for label, cmd in plan:
        print(f"[initium-setup] {label}: {shlex.join(cmd)}")
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[initium-setup] FAILED ({label}), exit code {exc.returncode}", file=sys.stderr)
            return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
