# Intel XPU 後續維護與正式版納入決策

決策日期：2026-10-19。來源：使用者於本次測試後明確指定。

## 維護基準

- 後續 Intel 更新以新的官方 XPU profile 為主要維護、相容性與驗收基準。
- 目前對應 `webui-user-intel-xpu-official.bat`、`venv-xpu-official`、
  PyTorch `2.14.1+xpu`／TorchVision `0.29.1+xpu`。日後套件更新沿此官方路徑評估，
  不是將以上版本永久鎖死。
- 舊 `webui-user-intel-arc.bat`／`venv-intel-arc`／PyTorch `2.14.0+xpu`
  路徑列為淘汰，不再作為後續 Intel 維護或測試主線。
- 使用者於同日進一步決定：**Kitchen 社群修改組暫停採用，移出本專案的啟動與建置支援範圍**；不再作為可選實驗推進。日後若要重新採用，另依使用者新決定處理。
- 本次停止社群支援，但不刪除使用者既有環境、模型或設定。舊 Intel 啟動檔保留為歷史相容入口，不代表仍列為主要支援路徑；Kitchen 專用啟動／建置檔已從測試分支移除。

## 分支與發布限制

- 新官方 Intel 路徑目前只停留在 `neo-test`。
- **等下一次大版本更新時，才將此 Intel 基準納入 `neo` 正式版。**
- 目前的測試版、一般修訂版或個別測試通過，不構成提前合併／發布的授權。
- 「下一次大版本」的具體版本號尚未指定，不能自行假設 `2.30-tw.2` 等修訂號
  就符合條件；也不能只因上游出現新版本就自動發布。
- 下一次大版本整合仍須保留原 A1111 布局、繁中、插件與 CUDA／XPU 環境隔離，
  並沿用 [UI Freeze](../UI_FREEZE.md) 規範與實機驗收。
- 本次不改 `neo`、不合併、不建立 Release、不變更程式版本識別。

## 決策時的實測狀態

官方 XPU／B580／test.6 使用原生 SDPA，可以完成 HR，第二次測試亦可完成，
但 VAE 階段仍觸發 OOM／Slice Attention 重試；使用者回報執行後整機嚴重卡頓。
截圖顯示專用顯存 11.1/12 GB、共享 GPU 記憶體 14.2/16 GB、RAM 89%。
這是「能用但仍有記憶體與體驗問題」，不是完整穩定驗收通過。

介面現有「Never OOM 記憶體最佳化 → 對VAE啟用（始終分塊）」可作下一步測試，
UNet 始終卸載保持不勾；本次尚未收到此設定的實測結果，不預先標記通過。
社群 Kitchen 組則因缺少原生 kernel 停在前置檢查，未完成生成測試。

## 同日追加：停止採用社群組

- 移除 `webui-user-intel-xpu-kitchen.bat` 與 `build-intel-xpu-kitchen.bat`。
- 移除 fork 原始碼下載、原生 kernel／wheel 建置安裝、專用 requirements 與驗收分支。
- `tools/launch_xpu_profile.py` 僅接受 `official`；舊的 `kitchen` profile／核心建置選項會在建立環境或安裝套件前拒絕，不靜默改跑官方組。
- 保留官方組 `venv-xpu-official`、`runtime/xpu-ab/official`、7862、Torch 2.14.1+xpu／TorchVision 0.29.1+xpu，避免重新初始化設定。
- **保留上游主程式仍需使用的 `comfy-kitchen==0.2.38`**；捨棄的是社群 XPU fork，不是刪除官方必要依賴。
- 不自動刪除使用者本機 `venv-xpu-kitchen` 或舊快取／輸出，亦不解除安裝開發工具。
- 歷史 review／changelog 保留原紀錄並註明撤回狀態；不修改已發布版本。
- 本輪完整 122 項 Python 測試通過，另驗證官方環境隔離、拒絕社群 profile、拒絕已撤回建置選項、既有版本資訊 fallback 與 UI Freeze；尚未新增 Windows／GPU 實機驗證。
