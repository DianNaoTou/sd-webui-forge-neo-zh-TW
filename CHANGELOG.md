# 更新紀錄

本專案的重要變更會記錄在此處。

## 2.30-tw.2-test.4（2026-10-19）

- 原 WD14 Tagger 介面新增 PixAI Tagger v0.9（官方連結之 DeepGHS ONNX）與 Camie Tagger v2（官方 ONNX），使用專用前處理與標籤解析，不替換舊模型載入器。
- 選取新模型、上傳圖片與啟動 WebUI 均不下載；首次按辨識／批次辨識或發出 API 辨識請求才下載。固定來源 revision，沿用 HF 快取，先查本機；已快取時可離線載入。
- 新模型下載／載入失敗顯示繁中錯誤並可重試，不影響其他模型；保留原標籤過濾、排序、批次與提示詞／圖片傳送流程。
- 不新增執行相依套件、不調整 PyTorch／TorchVision／ONNX Runtime 安裝或 CUDA／XPU 啟動檔；Intel Arc 沿用 CPU Tagger 後端。
- 新增 16 項測試，完整 107 項回歸通過；兩個新模型已完成實際權重 CPU 辨識、重複辨識及禁止網路後卸載重載。尚待 Windows／RTX 5060 Ti／B580 實機驗證。[完整說明](docs/reviews/2026-10-19-wd14-test4.md)。

## 2.30-tw.2-test.3（2026-10-18）

- 新增官方 XPU 與 Kitchen XPU 實驗組獨立啟動檔，兩組固定 torch 2.14.1+xpu／torchvision 0.29.1+xpu，以 Kitchen 為主要比較變數；保留原 Intel 2.14.0 啟動檔及 NVIDIA 環境。
- 分開 venv、設定、UI 設定、styles、輸出、HF／Torch／pip／暫存快取、連接埠及 Torch 安裝約束；只共用程式碼與模型路徑。擴充安裝設定階段序列化，避免 A/B 同時安裝衝突。
- 實驗組固定來源版本，提供原生建置入口與本機 Wheel／SHA256 安裝；檢查 Torch ABI、BMG package／AOT target、B580 裝置、XPU 能力與強制原生 INT8 smoke test，缺少配套時停止，不靜默回退冒充加速。
- Kitchen 修改版以獨立 requirements／安裝約束維持 0.2.37，官方組維持 0.2.38；不改根 requirements 或降級其他環境。
- 新增 12 項隔離／錯誤防護測試；[操作、建置條件與未驗證限制](docs/reviews/2026-10-18-xpu-ab-test3.md)。目前無公開配套 Windows 原生 Wheel，B 組須建置後實測；未宣稱兩組已通過 Windows／GPU 推論。

## 2.30-tw.2-test.2（2026-10-18）

- ControlNet 未選模型、模型不存在／無法辨識、已知 IPAdapter／一般 ControlNet 搭配錯誤時，明確提示並中止本次生成，避免悄悄退回無控制出圖。
- 初始化與採樣掛載失敗後清除單元參數，不再接續觸發 `KeyError: 0`；在 `process_images` 的 finally 執行可重複呼叫的請求清理，恢復掛載前 UNet。
- 修正一般 ControlNet 快取只保留權重、複本另建顯存包裝器的生命週期；快取與複本共用受管理包裝器，清理時保留記憶體管理器的追蹤，以便按需求卸載，不隱藏洩漏警告。
- 新增 CPU 回歸測試：錯配提前攔截、失敗中止與重試、多單元／HR 回呼、ADetailer 停用 ControlNet 的巢狀請求隔離、快取／複本生命週期與 dtype 分離。
- [完整修補說明與重測項目](docs/reviews/2026-10-18-controlnet-test2.md)。本輪尚無 GPU／Windows 修補後實測，不能宣稱記憶體洩漏已完全消除；正式 neo 維持不動。

## 2.30-tw.2-test.1（2026-10-18）

- 同步上游 `ce6848b8..f7f23149`，依實際 Diff 保留已整合 VAE 與我方 Flash 遮罩保護。
- Spectrum 重寫：分批／條件獨立預測、擬合快取；補上 HR 步數為 0 的回退、純量條件防護及預測串流讀取前等待。
- 修正權重錯誤分類、API 腳本預設參數重用，統一主模型／Refiner／HR／XYZ 模型清單刷新。
- 更新 Callback 呼叫者查找、設定頁排版、輸入框高度與 comfy-kitchen 0.2.38；保留本地依賴約束及繁中翻譯。
- Nunchaku 維持選用，移除自動降級 PyTorch；未確認相容 Wheel 的組合略過安裝，明確指定的 Wheel 以 `--no-deps` 安裝。
- [完整審查、衝突與驗證限制](docs/reviews/2026-10-18-upstream-sync.md)。此版未經 GPU 推論實測；正式版維持 2.30-tw.1。

## 2.30-tw.1（2026-10-13）

將已測試的 2.30-tw.1-test 升為正式版；完整新增功能、Bug 修正、驗證與已知限制見 [本次發布說明](docs/releases/2.30-tw.1.md)。

- 新增 Qwen-Image-2.1、DeGrid、正向／反向條件批次計算，以及 VAE／文字編碼器檔名顯示設定。
- 正式納入 NVIDIA CPU-only PyTorch 自動修復與 Intel XPU 後端防護；插件共用安裝流程約束 torch／torchvision 版本。
- 修正 DeGrid 寫死 CUDA、Qwen 2.1 分塊 VAE 四通道配置、Flash 遮罩在 Python -O 下被忽略，以及批次提示詞生成資訊。
- 補齊 2.30 新增繁中介面，修正提示詞延遲說明翻譯。
- 保留既有 XPU 查詢分塊與相依政策；未納入 ede60e1a causal Flash 優化，也未追加尚未測試的上游更新。
- B580 仍有 OUT_OF_DEVICE_MEMORY／DEVICE_LOST，本次未修復；依維護者決定列為已知限制後發布。
- 保留 2.29.2-tw.1／2.29.1-tw.1 固定 Tag 與 Release，另封存更新前正式分支。新增正式合併後的回歸與固定 Tag／Release 發布流程；neo-test 保持測試版標示。

## 2.30-tw.1-test（2026-10-11）

- 候選 PR：納入上游 `58700b99` 的 2D 分塊 VAE 解碼通道數修補，支援 Qwen 2.1 四通道輸出；新增數值回歸測試。
- 候選 PR：Flash Attention 有明確遮罩時保留 SDPA 分流，改用不受 Python `-O` 影響的保護；上游 `ede60e1a` causal 優化因套件版本／非方形對齊／任意遮罩風險保留待審。詳見 [技術審查](docs/reviews/2026-10-11-vae-flash.md)。
- `neo-test` 併入上游 Forge Neo 2.30（`ce6848b8`）：新增 Qwen-Image-2.1 支援、SDPA 注意力調整、DeGrid 腳本與多項小修；版本資訊改為 `neo 2.30-tw.1-test`。
- 保留本整合版修改：Intel XPU Math SDPA、Gradio 4.40／Pillow 10.4.0 相依組合、None／Ignore 提示詞權重修正與 SD 放大模型選單。
- 強化 PyTorch 後端防護（延續 Issue #4／`7e1b13e5`、`2007f401`）：Intel Arc 流程同樣偵測非 XPU 版 PyTorch 並自動重裝；重裝後仍非預期後端即停止啟動，不再改用 CPU；之後的插件安裝以已安裝的 torch／torchvision 版本為約束，避免被替換成 CPU 版。
- 修正上游新增的 DeGrid 後處理寫死 CUDA（`cf4f2e73`）：改用 Forge 偵測到的裝置（CUDA／XPU／CPU），該裝置失敗時自動改用 CPU，修正 Intel Arc 在「附加功能」處理圖片／影片時出錯；並新增測試，禁止主程式寫死 `.cuda()`。
- 補齊 2.30 新增介面的繁體中文翻譯（`85cdb821`）：DeGrid、Qwen 2.1 參考圖片、批次套用正向/反向調整、VAE／文字編碼器下拉選單顯示方式等。
- 修正「提示詞編輯傳送至 Gradio 前的延遲」說明翻譯未生效（`0feba725`），並新增語系檔格式、重複鍵與設定說明括號的檢查測試。
- 新增 CPU 環境可執行的回歸測試；尚待 RTX 與 Intel Arc 實機驗證，正式版仍為 `neo 2.29.2-tw.1`。

## 2.29.2-tw.1-test（2026-10-05）

- `neo-test` 的啟動與介面版本資訊改為 `neo 2.29.2-tw.1-test`，以 `-test` 後綴區分測試版。
- 本次僅調整版本標示與說明，生成行為及相依套件不變；正式版仍為 `neo 2.29.2-tw.1`。
- 正式發布前應移除 `-test`，並依當次更新調整上游版本及 `tw` 修訂號。

## 2.29.2-tw.1（2026-10-04）

本次正式分支更新由 2.29.1 升至 2.29.2-tw.1；上游版本為 2.29.2。更新內容、實測及已知限制見 [本次發布說明](docs/releases/2.29.2-tw.1.md)。

- 納入下方 2.29.2 測試階段的上游更新與相依修正。
- 修正 None／Ignore 提示詞權重模式回傳空值。
- XPU 查詢分塊後，使用者確認 B580 高解析度修復可完成，但速度較慢；後續曾遇 OUT_OF_RESOURCES／DEVICE_LOST，完整重啟後未再復現，尚未確認根因與連續生成穩定性。
- 使用者確認 RTX 5060 Ti 啟動與生成正常；XPU 修補後 NVIDIA 速度尚待重測。
- 新版版本資訊採用 2.29.2-tw.1，補上繁體中文版本與下載說明。
- 舊正式版 fecbe6b 以 2.29.1-tw.1 首個穩定版留存，不更動原程式。

## 2.29.1-tw.1 首個穩定版（2026-10-04 補記）

保存更新前正式提交 `fecbe6bf666a102680fcfe0626eae91a60fe0122`，程式仍顯示 2.29.1。整合版發布名稱與固定下載見 [保存紀錄](docs/releases/2.29.1-tw.1.md)。

## 歷史整合與測試紀錄

以下保留各項修改當時的狀態；正式發布時的最新驗證結果以上方發布說明為準。

### 2.29.2 測試版（2026-10-04）

- 上游同步至 [`d70373e`](https://github.com/Haoming02/sd-webui-forge-classic/commit/d70373ebcf1a96d210b78cd6f77196459e783e2a)，版本由 2.29.1 更新至 2.29.2。
- 修正 Klein、Krea2、Qwen Image 與 Z-Image 文字編碼器的單段／多段提示詞分詞處理，調整 SD／SDXL 分詞設定。
- Never OOM 的 UNet 強制卸載改為獨立控制；切換設定時只卸載相關 UNet，不再切換全域 VRAM 狀態或卸載所有模型。
- 依實際注意力實作與運算精度調整顯存需求估算。
- 將生成預覽更新移至採樣回呼，移除介面與 API 進度查詢中的重複更新。
- Refiner 卸載模型時加入找不到 UNet 的保護。
- 儲存 grouped INT8 量化權重時補齊 group_size 與 convrot_groupsize 參數。
- 減少 Lumina／SVDQ 不必要的張量複製操作；統一 Anima、Flux.2 與 Z-Image 的 Qwen 文字引擎屬性名稱。
- `comfy-kitchen` 更新至 0.2.37。
- 本整合版另外修正 Pillow、pillow-heif 與 protobuf 的依賴衝突，詳見「執行環境」。
- Intel XPU 的 Math SDPA 加入 query 分片，保留全長 K/V、遮罩與 GQA 行為，降低高解析度生成時完整注意力矩陣的記憶體尖峰；保留避免融合 SDPA 非有限輸出的相容性路徑。此為待 B580 實機驗證的候選修正。
- 驗證狀態：依賴政策測試 12 項通過；使用者回報 Windows／RTX 5060 Ti 啟動與生成正常，Intel Arc B580 啟動與基本生成正常，但高解析度修復第二輪採樣 OOM。XPU query 分片完成語法與 NumPy 模擬驗證，PyTorch 數值測試及 B580 高解析度修復重測仍待完成。本次僅更新測試分支，尚未發布為正式版。

### 新增

- 測試分支整合 Civitai Model Downloader v1.2.0，加入下載器、設定頁、狀態訊息與模型卡片操作的繁體中文翻譯。
- 補齊 LoRA 模型資訊視窗的訓練標籤、觸發詞、權重說明和卡片圖示提示文字。
- 首次安裝自動下載並驗證共用放大模型，提供下載進度條，且在內建放大、SD Upscale 與終極 SD 放大顯示繁中用途說明。
- 以 Forge Neo 最新 `neo` 分支為基礎建立 DianNaoTou 台灣繁體中文整合版。
- 加入完整 `zh_Hant` 語言檔，共 9,142 條介面翻譯。
- 全新安裝預設使用台灣繁體中文介面。
- 整合 ADetailer-Neo。
- 整合 forge-neo-image2prompt，包含本機相容性修正。
- 整合 sd-webui-prompt-all-in-one-neo，包含繁體中文翻譯。
- 整合 stable-diffusion-webui-tagger-fork，包含介面與相容性修正。
- 內建 Ultimate SD Upscale，保留原生 SD Upscale，並加入完整台灣繁體中文介面。
- 增加繁體中文 README，並保留上游英文原文。
- 新增 Intel Arc 通用啟動檔，沿用 Forge Neo 原始自動安裝流程並安裝 PyTorch XPU。
- NVIDIA 與 Intel Arc 使用獨立虛擬環境，避免 CUDA 與 XPU 套件互相覆蓋。
- Intel Arc 環境加入 XPU Math SDPA 相容性修正；Arc B580 為目前已驗證機型。
- 啟動器會以專案內的 `uv` 自動準備並鎖定 Python 3.13.12，不修改系統 Python、PATH 或 Registry。

### 上游同步

- 測試分支同步 Forge Neo 上游至 `e33f40e4`（2.29.1 開發分支）。
- LLLite ControlNet 配合 MultiDiffusion 分塊處理控制圖片；涵蓋 SDXL 與 Anima 路徑。
- 調整提示詞權重解析、LoRA Control、PNG Info 負面提示詞讀取及模型載入細節。
- 保留 `BREAK` 不算文字權重的判斷，避免 PNG Info 誤套用 Emphasis。
- MultiDiffusion 分塊設定加入固定介面識別碼；既有繁中翻譯可沿用。
- 補上 MultiDiffusion「從圖片自動偵測尺寸」按鈕提示的繁中翻譯。
- `README_EN.md` 更新至本次上游英文原文，繁中整合版說明維持於 `README.md`。
- 同步 Forge Neo 上游至 `41359cd4`。
- 新增提示詞輸入防抖與 img2img 保持長寬比模式。
- 整合模型載入、dtype、量化 metadata、RoPE、LoRA 與 ControlNet 修正。
- `comfy-kitchen` 更新至 0.2.35。

### 安全與容量

- 排除 Stable Diffusion、LoRA、VAE、Embedding、YOLO 等模型檔案。
- 排除使用者生成圖片、影片、`outputs`、`venv`、快取與備份檔。
- 插件所需模型仍依各插件原有機制在需要時下載。

### 執行環境

- 修正 2.29.2 上游合併後 requirements 與共用 constraints 衝突：沿用 Gradio 4.40 相容政策，統一 Pillow 10.4.0、pillow-heif 0.22.0 與 protobuf >=6.31.1,<7；新增跨檔鎖定一致性測試。

- 修正共用 `run_pip()` 將安裝專用參數附加到 `pip uninstall` 的問題，讓 CUDA／XPU 環境可正常移除衝突的 ONNX Runtime 套件。
- 跟隨 Forge Neo 最新上游需求。
- 目前上游測試環境為 Python 3.13.12。
- NVIDIA 使用 `venv-nvidia`，Intel Arc 使用 `venv-intel-arc`；兩者共用專案內的 Python 3.13.12 runtime。
- `uv` 下載支援斷點續傳、自動重試、Astral／GitHub 雙來源與 PowerShell 備援。
- NVIDIA 的 PyTorch／CUDA 套件由 Forge Neo 安裝器於首次啟動時自動安裝。
- Intel Arc 的 PyTorch XPU 套件由相同安裝流程於首次啟動時自動安裝。
