# 2.30-tw.2-test.1 上游整合審查

日期：2026-10-18（Asia/Taipei）；執行者：ChatGPT。
使用者授權直接整合／提交／推送測試分支並更新 ai-work-log；正式分支、Tag、Release 不動。

## 起始與範圍

- 正式 `neo`：`678453607fd97ae4a2a0f6d926631ce112d3161d`（2.30-tw.1）。
- 實際測試分支 `neo-test`：`1e1e4d6a704e950b90cae5acfdb8833865d9240f`（2.30-tw.1-test），不是 `test`。
- 上游 `Haoming02/sd-webui-forge-classic` / `neo`：`f7f231496b2dd46703f8da87e2e83b45a54ba73b`。
- Git merge-base／上次完整同步：`ce6848b819774ef23b21e115fa7e06de71ee5c05`。
- 比對範圍：`ce6848b8..f7f23149`，12 commits、21 檔。逐檔檢視實際 diff，再以 merge 保留歷史；不是按訊息盲目 cherry-pick。
- 原測試分支已含 `58700b99` 的 VAE 內容（PR #11），雖然沒有該上游 SHA，不重複套用。
- 正式分支後續只有發布文件、版本與 CI／發布流程差異；測試分支同步其目前版本文件，未帶入正式發布 workflow。

| Commit | 實際內容與處置 |
|---|---|
| 58700b99 | tiled VAE 三處 out_channels；原測試已具備，保留原內容 |
| ede60e1a | Qwen 傳 causal hint 納入；Flash 忽略 explicit mask 的改動不採用 |
| 9cae3d60 | LoRA Notes／metadata Description／XYZ 輸入框高度微調 |
| dde0f320 | safetensors 已知損壞錯誤轉為下載提示；權限／OOM 等其他例外保留原因 |
| af59a359 | 設定頁 grid、側欄選取樣式、搜尋框黏附、切頁捲動與按鈕名稱 |
| aa144fad | 模型／VAE 重掃，主刷新事件串接 Refiner、HR、XYZ，移除重複掃描按鈕 |
| 46af880c | API 讀取建立 UI 時記錄的 script.api_info.args，不重複呼叫 script.ui |
| 34022261 | Spectrum 分條件／批次預測器、擬合快取、stream、warmup／ready 判斷，允許 skip negative |
| 327b7f89 | comfy-kitchen 0.2.37 → 0.2.38 |
| f5292b0f | callback 以 frame 尋找呼叫者，取代整個 inspect.stack；其餘多為字串格式整理 |
| 04772d93 | Image Stitch 編輯模型提示使用 tooltip，補 Qwen-Image-2.1 |
| f7f23149 | 隱藏 comfy_attention 相關 unexpected weights 訊息；不忽略實際 missing weights |

## 衝突與本地修正

真正 Git 衝突僅兩處：

1. `backend/attention.py`：採我方 explicit mask → SDPA 分流，保留不受 Python `-O` 移除的防護。不無條件相信 causal hint、不捨棄 mask。Flash 無 mask 與失敗回退保留；本輪沒有實作 causal Flash 加速。既有風險分析見 [10/11 審查](2026-10-11-vae-flash.md)。
2. `extensions-builtin/sd_forge_lora/ui_edit_user_metadata.py`：保留「備註」與其他既有中文，只把 lines 改成 1.05。

Spectrum 另外修正三處可由程式確認的相容問題：

- HR steps=0 的含義是繼承 p.steps；上游直接使用 0 會使 warmup／stop 與時間正規化不正確，現回退原步數。
- 條件包含 0 維 tensor 時不可讀 shape[0]，現先檢查 ndim。
- forecast stream 消費模型輸出前，先等待目前 stream；原本只在結尾讓目前 stream 等待 forecast，缺少讀取方向的順序。GPU 串流行為仍須實測。

`localizations/zh_Hant.json` 保留全部舊翻譯，加 Show All Pages／Show Single Page／Edit Models。
版本檔、README、RELEASES、CHANGELOG 與本說明統一標示 2.30-tw.2-test.1；歷史版號保持歷史意義。

## Nunchaku

2026-10-18 實際查詢官方 release API，原 nunchaku-ai/nunchaku 重新導向 nunchux-ai/nunchaku：

- [v1.2.1](https://github.com/nunchux-ai/nunchaku/releases/tag/v1.2.1) 有 56 個 wheel；Windows／Linux、cp310–cp313。
- cu12.8：torch 2.8／2.9／2.10／2.11；cu13.0：torch 2.9／2.10／2.11。
- 沒有 torch2.13 wheel。查到的前四個 1.3.0 開發 release 亦無 torch2.13 資產；不能以 CUDA 13.0 支援推論 PyTorch 2.13 ABI 相容。
- 移除既有 launcher 中 `--nunchaku` 觸發 PyTorch 2.11 降級的分支。
- 保留選用開關；只為已確認資產矩陣產生預設 URL。不支援組合跳過安裝並顯示原因。
- 保留 `NUNCHAKU_PACKAGE` 手動指定來源；安裝使用 `--no-deps`，並保留現有 torch constraints，避免替換整個環境。自行指定 wheel 的 ABI 與額外依賴由使用者確認；不宣稱該自訂來源已通過驗證。
- Intel XPU 仍不允許 CUDA-only 選項；沒有嘗試安裝 Nunchaku。

[comfy-kitchen 0.2.38 官方 metadata](https://pypi.org/pypi/comfy-kitchen/0.2.38/json) 具 Windows wheels、Python >=3.10，基本 requires_dist 沒有強制替換 torch。可選 cublas extra 未啟用。這不等於已驗證 GPU kernel。

## 本地功能保存依據

與起始 neo-test 比對，以下逐檔完全未變：`backend/attention.py`、`backend/xpu_attention.py`、`backend/patcher/vae.py`、`scripts/degrid.py`、`constraints-common.txt`、CUDA／Intel Arc 啟動批次檔、`extensions/` 全部檔案（含 WD14 Tagger／ADetailer-Neo 等）。
`modules/ui.py` 保留我方 upscale 選單內容，只納入上游刷新機制；`requirements.txt` 僅 comfy-kitchen 版本改變，Gradio 4.40／Pillow 10.4／transformers／ONNX 等本地約束未變。沒有刪除追蹤檔案。
這是原始碼與 CPU 回歸驗證，不能替代插件實際操作／GPU／Windows 啟動測試。

## 檢查結果

Linux、Python 3.12、PyTorch 2.14.1+cpu（隔離的工作機測試依賴，未變更專案目標 torch）。

- `python -m unittest discover -s tests -v`：66 tests 全通過，0 skip。
- `python -O -m unittest discover -s tests -p test_vae_flash_regressions.py -v`：5 tests 全通過。
- 新增 6 tests：Nunchaku ABI 安裝矩陣／不降級、API 預設值含 False/0 且不重建 UI、刷新 callback 串接與 UI reload 重設、Spectrum 多批次 CFG／中途跳過負面／新 pass、HR 0 步數繼承、權重錯誤分類。
- Spectrum 測試載入真實 forecaster 與 wrapper，僅將裝置初始化替換成 CPU；不宣稱 GPU stream 已測。
- `compileall` 核心、tests、全部 extensions-builtin 成功；舊 annotator 有既存 invalid escape SyntaxWarning，非本輪修改。
- `node --check javascript/settings.js`、`git diff --check` 成功。
- 既有回歸含 VAE 3/4 通道、Qwen 真實 mask builder、Flash 模擬 kernel fallback、XPU 數值、安裝與擴充整合／語系檢查。
- 本工作機 `pip check` 通過；另由 regression 檢查專案 requirements 與 constraints 相容。不表示已在本工作機安裝全部 WebUI 依賴。
- 初次測試因工作機缺 tqdm 失敗，補上測試依賴後重跑成功。Spectrum 常數訊號測試初次以正則化 0.1 搭配過緊的精確值容差而失敗；測試改用微小正則化 1e-5，以檢查歷史索引／分流；產品預設正則化不變。
- CPU workflow 增加 neo-test push 觸發並編譯 Spectrum；遠端結果另記於工作 Issue。

未執行：完整 WebUI／API server 啟動、真實模型載入、Windows bat 安裝、CUDA／XPU kernel、瀏覽器操作、插件生成、Nunchaku wheel 二進位載入。

## 使用者本機驗證清單

兩台都先確認目前分支 neo-test，關閉 WebUI 後 `git pull --ff-only` 再完整重啟；版本應為 neo 2.30-tw.2-test.1。

| 項目 | RTX 5060 Ti 16GB | Arc B580 12GB |
|---|---|---|
| 啟動與環境 | torch 2.13 +cu130，無 CPU 回退 | 原 +xpu 不被替換，無 CUDA-only 選項 |
| 基本生成 | SDXL 相同 seed、尺寸、步數，比對 Spectrum 關閉的結果 | 相同流程，檢查 NaN、黑圖、device error |
| Spectrum | 開／關比對耗時與品質；batch 1/2、Batch Cond/Uncond、skip negative 開關 | 先小尺寸，再相同組合；檢查 dtype／stream／OOM |
| HR | HR steps=0 與明確 10；相同／不同 checkpoint、VAE、TE | 倍率先 1.5；記錄速度與峰值用量 |
| Qwen-Image 2.1 | 生成／參考圖／tiled VAE；有 Flash 時仍保留 mask | 有足夠資源才測；OOM 與模型／步數分開記錄 |
| 模型刷新 | 新增模型後點主刷新，Refiner／HR／XYZ 選單同步；切換模型 | 同左 |
| 擴充／後處理 | WD14 傳送、ADetailer、Upscalers、DeGrid | 同左，確認後處理無 CUDA 寫死 |
| API | --api 與 --nowebui 各驗證預設腳本參數及一張 txt2img／img2img | 同左（可行時） |
| 穩定性與語系 | 連續生成至少 3 次、設定頁切頁與新文字 | 同左；遇舊 OUT_OF_RESOURCES／DEVICE_LOST 保留完整 log |

Nunchaku 不列為本版必要生成測試，預設不開啟；torch2.13 無已確認 wheel 時應只顯示跳過，不降級。

已知風險：Spectrum 是近似加速，畫質／速度／記憶體可能隨模型與參數改變；新實作最多為每個條件／batch 保留 100 筆歷史，需特別觀察 B580 顯存。Flash causal 加速刻意未採用；此版不代表既有 B580 連續生成資源錯誤已解決。
