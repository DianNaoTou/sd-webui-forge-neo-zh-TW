# Stable Diffusion WebUI Forge Neo
## DianNaoTou 台灣繁體中文整合版

<p align="center">
  <a href="./README.md"><b>繁體中文</b></a> ｜ <a href="./README_EN.md">English（官方原文）</a>
</p>

<p align="center"><img src="html/ui.webp" width="512" alt="Stable Diffusion WebUI Forge Neo 介面"></p>

> [!IMPORTANT]
> 本專案是由 **DianNaoTou** 維護的非官方台灣繁體中文整合版，以
> [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
> 的 `neo` 分支為基礎。Forge Neo 與各插件的著作權仍屬原作者所有。

## 專案介紹

Stable Diffusion WebUI Forge Neo 是以 AUTOMATIC1111 Stable Diffusion WebUI 為基礎的圖像生成介面，著重於模型相容性、顯存管理、推論效能與易用性。

這個整合版的目標，是提供一套可直接安裝、同時維護 **NVIDIA** 與
**Intel Arc** 執行環境的台灣繁體中文 Forge Neo：

- 完整台灣繁體中文介面。
- 補齊 Forge Neo 設定頁面的翻譯。
- 整合常用插件及其繁體中文化修正。
- 保留原專案的安裝與自動下載機制。
- 依顯示卡選擇啟動檔，首次啟動會自動準備專案專用 Python、建立環境並安裝對應的 PyTorch。
- Intel Arc 使用獨立虛擬環境及 XPU 相容性修正，不影響 NVIDIA 環境。
- 不附帶任何生成模型或使用者輸出內容。

## 整合內容

| 元件 | 用途 | 整合狀態 |
|---|---|---|
| Forge Neo | Stable Diffusion WebUI 主程式 | 基於官方 `neo` 分支 |
| ADetailer-Neo | 自動偵測、遮罩與局部重繪 | 整合繁體中文介面 |
| forge-neo-image2prompt | 圖片反推提示詞 | 包含相容性修正 |
| sd-webui-prompt-all-in-one-neo | 提示詞管理與編輯 | 包含繁體中文翻譯 |
| stable-diffusion-webui-tagger-fork | 圖片標籤反推 | 包含介面與相容性修正 |
| Ultimate SD Upscale（終極 SD 放大） | 高解析度分塊放大與重繪 | 內建並整合繁體中文介面 |
| zh_Hant 語言檔 | Forge Neo 與插件介面翻譯 | 由 DianNaoTou 整合維護 |

實際收錄版本與變更內容會記錄於 `CHANGELOG.md`。

## 不包含的內容

為避免授權、容量與隱私問題，本儲存庫不包含：

- Stable Diffusion Checkpoint。
- LoRA、VAE、Embedding。
- YOLO 與其他偵測模型。
- 使用者生成的圖片及影片。
- `outputs`、`models`、`venv` 與快取資料夾。
- 個人提示詞、瀏覽器資料及本機絕對路徑。

ADetailer-Neo 等插件需要的偵測模型，會按照插件原有機制在需要時自動下載。Stable Diffusion 主模型請使用者自行準備。

## 顯示卡支援

| 平台 | 啟動檔 | PyTorch | 狀態 |
|---|---|---|---|
| NVIDIA | `webui-user.bat` | Forge Neo 上游預設 CUDA 版本 | 跟隨上游維護 |
| Intel Arc | `webui-user-intel-arc.bat` | PyTorch XPU | 整合 XPU Math SDPA 相容性修正 |

兩種啟動方式共用同一套 Forge Neo、繁體中文、內建插件與模型目錄，但使用不同的虛擬環境，避免 CUDA 與 XPU 套件互相覆蓋。

Intel Arc B580 是目前實際驗證的 Intel 機型；Intel 啟動方式並未寫死 B580 型號，其他 Arc 顯示卡仍取決於驅動程式與 PyTorch XPU 的硬體支援。

## Windows 安裝方式

### 需求

- Windows 10／11
- Git
- NVIDIA：支援上游目前 CUDA 版本的驅動程式
- Intel Arc：近期 Intel Graphics Driver

不需要另外安裝或更改系統 Python。啟動檔會在專案的 `runtime` 目錄內自動準備
Python 3.13.12；電腦原有的 Python、PATH 與 Windows Registry 都不會被修改。

### 下載專案

在命令提示字元中執行：

```bat
git clone https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git
cd sd-webui-forge-neo-zh-TW
```

### NVIDIA

```bat
webui-user.bat
```

使用獨立的 `venv-nvidia`，並由 Forge Neo 安裝器自動安裝上游預設的 PyTorch 與 CUDA 套件。

### Intel Arc

```bat
webui-user-intel-arc.bat
```

使用獨立的 `venv-intel-arc`，並由 Forge Neo 安裝器自動安裝 PyTorch XPU 與其他相依套件。完整說明請參閱 [`INTEL_ARC_SETUP.md`](./INTEL_ARC_SETUP.md)。

兩種版本都保留 Forge Neo 原本的安裝體驗：首次啟動會先以專案內的 `uv` 準備固定版本 Python 3.13.12，再建立對應的虛擬環境，並由 Forge Neo 安裝器自動安裝 PyTorch、核心相依套件與插件相依套件。它不會修改系統 Python，也不會自動安裝顯示卡驅動程式或 Stable Diffusion 主模型。

第一次啟動需要網路連線，下載內容會存放在專案的 `runtime`、`venv-nvidia` 或
`venv-intel-arc` 目錄。日後啟動會直接重用；若曾經用錯誤的系統 Python 建立舊
`venv`，新版啟動檔也不會使用或刪除它。

本整合版目前跟隨 Forge Neo 最新 `neo` 分支，不鎖定舊版核心；上游更新可能改變 Python、PyTorch 或 CUDA 需求。

> [!NOTE]
> 本專案不會下載 Stable Diffusion 主模型。請自行將模型放入對應的 `models` 子目錄，或設定外部模型路徑。

## 繁體中文介面

整合版會提供完整的 `zh_Hant` 語言檔，涵蓋：

- Forge Neo 主要頁面。
- 設定頁面與進階選項。
- ADetailer-Neo。
- Prompt All-in-One Neo。
- Tagger 與 Image2Prompt 等已整合插件。

若介面沒有自動切換，可前往：

**設定 → 使用者介面 → 語言／Localization**

選擇繁體中文後套用設定並重新載入介面。

## 終極 SD 放大（Ultimate SD Upscale）

本專案已內建終極 SD 放大，可在圖生圖的「腳本」選單直接使用，不必另外安裝擴充。它適合高解析度圖片放大與分塊式 img2img 重繪，能降低一次處理超大圖片造成的 VRAM 壓力，並比原生 SD Upscale 提供更多分塊與接縫控制。原生 SD Upscale 仍完整保留。

## 共用放大模型

本專案不把大型模型檔提交進 Git；NVIDIA 與 Intel Arc 的首次安裝流程會共用同一套下載器，將下列模型放入 `models\ESRGAN`：

- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_anime_6B.pth`
- `4x-AnimeSharp.pth`
- `4x-UltraSharp.pth`
- `4x_foolhardy_Remacri.pth`

五個模型合計約 274 MB，只會下載缺少的檔案。

已存在的同名模型會直接跳過且不覆寫；新下載檔會先以 SHA-256 驗證，再以完整檔名啟用。單一模型下載失敗只會顯示警告，不會阻止 Forge Neo 啟動，之後可在專案根目錄執行 `download-upscalers.bat` 重試。

這些模型只儲存一份，並由內建圖片放大、圖生圖／SD Upscale、Hires. fix 與終極 SD 放大共用。介面保留原始模型識別值，並在選單中加上繁中用途提示。

## 更新與上游同步

本專案會盡量同步 Forge Neo 與各插件的重要更新，但不保證與上游同日更新。

由於本整合版包含中文化及相容性修改，直接使用插件內建更新功能，可能會覆蓋部分修改。更新前建議先備份，並查看本專案的更新紀錄。

## 回報問題

回報問題時，請盡量提供：

- Windows 版本。
- 顯示卡型號與顯存容量。
- 啟動參數。
- 錯誤訊息或主控台紀錄。
- 可以重現問題的操作步驟。

請勿上傳含有個人資料、私人圖片、API 金鑰或登入憑證的檔案。

## 專案來源與致謝

本整合版建立在眾多開源專案與社群翻譯成果之上，特別感謝：

- [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Haoming02/ADetailer-Neo](https://github.com/Haoming02/ADetailer-Neo)
- [Adeliox/forge-neo-image2prompt](https://github.com/Adeliox/forge-neo-image2prompt)
- [eduardoabreu81/sd-webui-prompt-all-in-one-neo](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo)
- [Kataragi/stable-diffusion-webui-tagger-fork](https://github.com/Kataragi/stable-diffusion-webui-tagger-fork)
- [Coyote-A/ultimate-upscale-for-automatic1111](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)
- [bluelovers/stable-diffusion-webui-localization-zh_Hant](https://github.com/bluelovers/stable-diffusion-webui-localization-zh_Hant)
- 所有原始翻譯作者、插件維護者與測試使用者。

**繁體中文化與整合維護：DianNaoTou**

## 授權

Forge Neo 主程式及各插件分別依其原始授權發布。本整合版保留上游授權檔、著作權聲明及來源資訊；個別元件的使用與再散布條件，請以其原始授權為準。

---

如果這個整合版對你有幫助，歡迎提交 Issue、協助測試或改善翻譯。
