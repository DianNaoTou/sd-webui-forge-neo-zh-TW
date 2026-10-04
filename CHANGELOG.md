# 更新紀錄

本專案的重要變更會記錄在此處。

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
