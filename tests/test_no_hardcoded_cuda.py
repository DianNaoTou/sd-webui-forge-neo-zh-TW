"""Guard against hard-coded CUDA in core WebUI code paths.

Intel Arc (XPU) and CPU users must not hit `.cuda()` in scripts/ or
modules/. Use Forge's selected device (modules.devices.device) instead.
Upstream DeGrid (scripts/degrid.py) originally called image.cuda(),
which broke Extras post-processing on XPU.
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKED_DIRS = ("scripts", "modules", "modules_forge")
PATTERN = re.compile(r"\.cuda\(\)")


class NoHardcodedCudaTests(unittest.TestCase):
    def test_core_code_has_no_bare_cuda_calls(self):
        offenders = []
        for d in CHECKED_DIRS:
            for path in sorted((ROOT / d).rglob("*.py")):
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    code = line.split("#", 1)[0]
                    if PATTERN.search(code):
                        offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "Hard-coded .cuda() found; use modules.devices.device:\n" + "\n".join(offenders))

    def test_degrid_uses_forge_device_with_cpu_fallback(self):
        src = (ROOT / "scripts" / "degrid.py").read_text(encoding="utf-8")
        self.assertIn("devices.device", src)
        self.assertIn("devices.cpu", src)


if __name__ == "__main__":
    unittest.main()
