# Intel XPU 後續維護與正式版納入決策

決策日期：2026-10-19。來源：使用者於本次測試後明確指定。

## 維護基準

- 後續 Intel 更新以新的官方 XPU profile 為主要維護、相容性與驗收基準。
- 目前對應 `webui-user-intel-arc.bat`（同日統一入口後）、`venv-xpu-official`、
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

後續使用者實測：開啟「Never OOM 記憶體最佳化 → 對VAE啟用（始終分塊）」後，
原先炸機／卡頓情況明顯改善，並認為此環境用起來穩定；這是使用者的 B580 回報，
不是所有模型、解析度或其他 Arc 型號的完整驗收。UNet 始終卸載保持不勾。
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

## 同日追加：官方 XPU 直接取代舊 Intel Arc 入口

使用者確認新版體驗穩定，授權直接捨棄舊 Intel Arc 路線。
`webui-user-intel-arc.bat` 改為轉交 `webui-user-intel-xpu-official.bat`，傳遞參數與結束狀態。
不再安裝 Torch 2.14.0 或建立／啟用 `venv-intel-arc`；兩入口共用既有
`venv-xpu-official`、`runtime/xpu-ab/official` 與 7862，避免重新初始化已測好的設定。
既有舊 venv、根目錄設定與資料保留，不自動搬移或刪除。原先「保留歷史入口」的決策
由此次轉交行為取代；正式分支下一次大版本才整合的限制不變。

## 同日最終簡化：僅保留 Intel Arc 一個入口

依使用者要求，官方 XPU 啟動流程直接置入 `webui-user-intel-arc.bat`，
移除 `webui-user-intel-xpu-official.bat`。本節取代上一節的雙入口轉交安排。
官方 Python launcher、額外參數、失敗處理與環境／資料路徑完全沿用，
不重新建立 venv、不搬移或改寫既有設定。正式分支與發布限制維持不變。


## 2026-10-25：2.31 正式版發布授權

使用者完成簡單測試後，明確要求將 Intel WD14 GPU 修改前的 Test 3 升至正式分支。正式版號為 `2.31-tw.1`，本次授權使官方 XPU 基準隨 2.31 大版本正式納入；上文 10-19 當時的暫緩狀態保留為歷史紀錄。本次不包含 Test 4 的 OpenVINO worker、GPU 啟動檔或相關依賴修改。B580 既有 OOM／DEVICE_LOST 限制保留，未宣稱全面修復。


## 2026-10-25：Test 5 統一 WD14 GPU 啟動入口

使用者回報 Test 4 專用 GPU 啟動檔在 B580 功能正常，明確要求以該流程取代原 Intel Arc 啟動方式，捨棄雙 BAT 選擇。本次僅更新 `neo-test`、發布 `2.31-tw.1-test.5`：保留 `webui-user-intel-arc.bat` 名稱但改為 GPU 準備入口，刪除重複的 `webui-user-intel-arc-wd14-gpu.bat`，保留失敗時 CPU 自動退回及既有官方 XPU profile。NVIDIA 與正式 `neo` 不變。
