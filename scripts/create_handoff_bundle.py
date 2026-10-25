from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "handoff_bundle"
ZIP_PATH = PROJECT_ROOT / "PyFly_handoff_bundle.zip"

INCLUDE_DIRS = [
    "src",
    "scripts",
    "configs",
    "tests",
    "data/raw",
    "data/processed",
    "data/mappings",
    "artifacts/metrics",
    "artifacts/models",
]

INCLUDE_ROOT_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "README.md",
]

FEATURE_DIR = PROJECT_ROOT / "artifacts" / "features"


def run_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as exc:
        return {
            "command": command,
            "error": repr(exc),
        }


def copy_tree(relative_path: str) -> None:
    source = PROJECT_ROOT / relative_path

    if not source.exists():
        return

    if source.is_file():
        destination = BUNDLE_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        if "__pycache__" in path.parts:
            continue

        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue

        relative = path.relative_to(PROJECT_ROOT)
        target = BUNDLE_DIR / relative

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            path,
            target,
        )


def build_file_inventory() -> list[str]:
    ignored = {
        ".venv",
        ".idea",
        "__pycache__",
        "handoff_bundle",
    }

    rows = []

    for path in sorted(PROJECT_ROOT.rglob("*")):
        relative = path.relative_to(PROJECT_ROOT)

        if any(part in ignored for part in relative.parts):
            continue

        if "fly-data" in relative.parts:
            continue

        if not path.is_file():
            continue

        try:
            size = path.stat().st_size
        except OSError:
            size = -1

        rows.append(
            f"{relative} | {size} bytes"
        )

    return rows


def feature_inventory() -> list[dict]:
    rows = []

    if not FEATURE_DIR.exists():
        return rows

    for path in sorted(FEATURE_DIR.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                }
            )

    return rows


def main() -> None:
    # Remove previous bundle folder if it exists
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)

    # Remove previous ZIP if it exists
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    # Create fresh bundle directory
    BUNDLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("============================================")
    print("       CREATING PYFLY HANDOFF BUNDLE")
    print("============================================")
    print()

    # ---------------------------------------------------------
    # Copy project directories
    # ---------------------------------------------------------

    print("Copying project directories...")

    for relative in INCLUDE_DIRS:
        print(f"  {relative}")
        copy_tree(relative)

    # ---------------------------------------------------------
    # Copy root files
    # ---------------------------------------------------------

    print()
    print("Copying root files...")

    for filename in INCLUDE_ROOT_FILES:
        print(f"  {filename}")
        copy_tree(filename)

    # ---------------------------------------------------------
    # Collect environment / machine information
    # ---------------------------------------------------------

    print()
    print("Collecting environment information...")

    manifest = {
        "created_at": datetime.now().isoformat(),

        "project_root": str(PROJECT_ROOT),

        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },

        "commands": {
            "pip_freeze": run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "freeze",
                ]
            ),

            "flybrain_info": run_command(
                [
                    sys.executable,
                    "-m",
                    "flybrain",
                    "info",
                ]
            ),

            "nvidia_smi": run_command(
                [
                    "nvidia-smi",
                ]
            ),
        },

        "feature_inventory": feature_inventory(),
    }

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

    manifest_path = (
        BUNDLE_DIR
        / "HANDOFF_MANIFEST.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Created: {manifest_path.name}"
    )

    # ---------------------------------------------------------
    # Save project file inventory
    # ---------------------------------------------------------

    inventory_path = (
        BUNDLE_DIR
        / "PROJECT_FILE_INVENTORY.txt"
    )

    inventory_path.write_text(
        "\n".join(
            build_file_inventory()
        ),
        encoding="utf-8",
    )

    print(
        f"Created: {inventory_path.name}"
    )

    # ---------------------------------------------------------
    # Save Git status
    # ---------------------------------------------------------

    git_status_path = (
        BUNDLE_DIR
        / "GIT_STATUS.json"
    )

    git_status_path.write_text(
        json.dumps(
            run_command(
                [
                    "git",
                    "status",
                    "--short",
                ]
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Created: {git_status_path.name}"
    )

    # ---------------------------------------------------------
    # Save Git diff
    # ---------------------------------------------------------

    git_diff_path = (
        BUNDLE_DIR
        / "GIT_DIFF.json"
    )

    git_diff_path.write_text(
        json.dumps(
            run_command(
                [
                    "git",
                    "diff",
                ]
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Created: {git_diff_path.name}"
    )

    # ---------------------------------------------------------
    # Create ZIP archive
    # ---------------------------------------------------------

    print()
    print("Creating ZIP archive...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for path in BUNDLE_DIR.rglob("*"):

            if not path.is_file():
                continue

            archive.write(
                path,
                arcname=path.relative_to(
                    BUNDLE_DIR
                ),
            )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("============================================")
    print("       PYFLY HANDOFF BUNDLE COMPLETE")
    print("============================================")
    print()

    print(
        f"Bundle folder: {BUNDLE_DIR}"
    )

    print(
        f"ZIP:           {ZIP_PATH}"
    )

    print()

    if ZIP_PATH.exists():

        zip_size_mb = (
            ZIP_PATH.stat().st_size
            / 1024
            / 1024
        )

        print(
            f"ZIP size:      "
            f"{zip_size_mb:.2f} MB"
        )

    print()

    print(
        "Upload PyFly_handoff_bundle.zip "
        "in the new chat."
    )

    print()


if __name__ == "__main__":
    main()