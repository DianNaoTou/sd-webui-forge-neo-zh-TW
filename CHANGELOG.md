# 更新紀錄

本專案的重要變更會記錄在此處。

## 尚未發布

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

- 修正共用 `run_pip()` 將安裝專用參數附加到 `pip uninstall` 的問題，讓 CUDA／XPU 環境可正常移除衝突的 ONNX Runtime 套件。
- 跟隨 Forge Neo 最新上游需求。
- 目前上游測試環境為 Python 3.13.12。
- NVIDIA 使用 `venv-nvidia`，Intel Arc 使用 `venv-intel-arc`；兩者共用專案內的 Python 3.13.12 runtime。
- `uv` 下載支援斷點續傳、自動重試、Astral／GitHub 雙來源與 PowerShell 備援。
- NVIDIA 的 PyTorch／CUDA 套件由 Forge Neo 安裝器於首次啟動時自動安裝。
- Intel Arc 的 PyTorch XPU 套件由相同安裝流程於首次啟動時自動安裝。
