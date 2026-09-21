# Upstream tracking

- Project: `Coyote-A/ultimate-upscale-for-automatic1111`
- Source: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111
- Branch: `master`
- Vendored commit: `2322caa480535b1011a1f9c18126d85ea444f146`
- Commit date: 2024-03-08
- License: GPL-3.0 (see `LICENSE`)

The extension is vendored under `extensions-builtin` so a normal clone is ready
to use without Git submodules or a network download during first startup.  Keep
the upstream files unmodified whenever possible; Traditional Chinese UI text is
maintained in this extension's `localizations/zh_Hant.json` file.

To update, compare the recorded commit with upstream, review the upstream diff,
replace the vendored `README.md`, `LICENSE`, and `scripts/ultimate-upscale.py`,
then update this commit record and run the repository tests.
