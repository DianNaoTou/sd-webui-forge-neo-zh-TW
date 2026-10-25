# 來源與修改紀錄

- 官方：https://github.com/lllyasviel/sd-forge-layerdiffuse
- 固定官方提交：`b1e66511e3a405a9e671da0755fc5356e033e97f`
- 修正討論：https://github.com/lllyasviel/sd-forge-layerdiffuse/issues/153
- Fix 3：https://github.com/user-attachments/files/32383696/sd-forge-layerdiffuse.for.Forge.Neo.2026.fix.3.zip
- 受審 ZIP SHA-256：`fdee87521680c6d31d4820840012929436cddc557c535b59b34cbcf9943de1a8`
- 上游授權：Apache-2.0；隨附原 LICENSE，原有作者註記保留。

## DianNaoTou 整合修改，2026-10-23

以 Fix 3 的五個 Python 原始碼及 LICENSE 為基礎，未納入附件 `.git` 或 `.pyc`。
修改 `utils.py` 的 WeightAdapter 格式、通道擴充限制、完整套用檢查；保留先 detach 再移除 clone 的修正。
修改 `attention_sharing.py` 的相容卸載呼叫，移除註解掉的除錯輸出。
修改 `forge_layerdiffusion.py` 的繁中介面、驗證、請求範圍快取、採樣條件與 finally 清理；不寫入全域 is_inpaint。
新增 `compat.py`、`downloads.py`、`weights.json`，不改 Forge 核心，不新增外部依賴。
`models.py` 與 `enums.py` 邏輯沿用上游，僅統一文字換行。

權重來源：`LayerDiffusion/layerdiffusion-v1`，固定 revision `e337e9d7a0f2a6fee6c3580374327380f88d065b`。
`weights.json` 的 SHA-256／size 來自該固定 revision 的官方 Hugging Face tree API LFS 中繼資料；整合時未下載權重，並非聲稱已實測權重內容。

完整限制及用法見 README.md；本機回歸測試不取代 GPU／UI 實機驗證。
