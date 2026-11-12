"""Optional, reversible compatibility patch for sd-forge-couple v7.1.0.

Run from the Forge Neo repository root:
    python tools/forge_couple_prompt_compat.py --check
    python tools/forge_couple_prompt_compat.py --apply
    python tools/forge_couple_prompt_compat.py --restore

Stop Forge Neo before applying/restoring. This does not install the extension.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "extensions" / "sd-forge-couple" / "scripts" / "forge_couple.py"
BACKUP = TARGET.with_suffix(".py.before-dnt-fc-patch")

OLD = '''        prompts: str = kwargs["prompts"][0]

        if common_parser in ("{ }", "< >"):
'''
NEW = '''        # DNT optional compatibility fallback (see tools/forge_couple_prompt_compat.py).
        # Some prompt-processing extensions transform kwargs["prompts"][0]
        # before Forge Couple receives it. Use a better-preserved original
        # only when it contains strictly more regional separators.
        prompts: str = kwargs["prompts"][0]
        prompt_sources = [
            ("kwargs", prompts),
            ("p.prompt", getattr(p, "prompt", None)),
            ("p.all_prompts[0]", (
                getattr(p, "all_prompts", [None])[0]
                if getattr(p, "all_prompts", None) else None
            )),
        ]
        def region_count(value):
            return len(value.split(separator)) if isinstance(value, str) else 0
        source, selected = max(
            prompt_sources, key=lambda pair: region_count(pair[1])
        )
        if region_count(selected) > region_count(prompts):
            logger.info(
                f"Prompt compatibility fallback: {source} "
                f"({region_count(prompts)} -> {region_count(selected)} regions)"
            )
            prompts = selected

        if common_parser in ("{ }", "< >"):
'''
# Requires upstream's ordering: the separator has already been normalized above.
# Preserve the user's separator choice; do not automatically rewrite prompts.

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if not TARGET.is_file():
        print(f"Forge Couple not installed at: {TARGET}")
        return 2
    if args.restore:
        if not BACKUP.is_file():
            print(f"Backup not found: {BACKUP}")
            return 2
        shutil.copy2(BACKUP, TARGET)
        print("Restored Forge Couple from backup. Restart Forge Neo.")
        return 0

    original = TARGET.read_text(encoding="utf-8")
    if NEW in original:
        print("Patch already installed.")
        return 0
    if original.count(OLD) != 1:
        print("Unsupported Forge Couple version. No files changed.")
        return 2
    if args.check:
        print("Forge Couple v7.1-style patch point found; safe to apply.")
        return 0
    if BACKUP.exists():
        print(f"Existing backup at {BACKUP}; refusing overwrite. No files changed.")
        return 2
    shutil.copy2(TARGET, BACKUP)
    try:
        updated = original.replace(OLD, NEW, 1)
        compile(updated, str(TARGET), "exec")
        TARGET.write_text(updated, encoding="utf-8")
    except Exception:
        shutil.copy2(BACKUP, TARGET)
        raise
    print("Patch applied. Restart Forge Neo; test with proper regional separators.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
