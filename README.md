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

這個整合版的目標，是提供一套可直接安裝的 **台灣繁體中文 Forge Neo**：

- 完整台灣繁體中文介面。
- 補齊 Forge Neo 設定頁面的翻譯。
- 整合常用插件及其繁體中文化修正。
- 保留原專案的安裝與自動下載機制。
- 不附帶任何生成模型或使用者輸出內容。

## 整合內容

| 元件 | 用途 | 整合狀態 |
|---|---|---|
| Forge Neo | Stable Diffusion WebUI 主程式 | 基於官方 `neo` 分支 |
| ADetailer-Neo | 自動偵測、遮罩與局部重繪 | 整合繁體中文介面 |
| forge-neo-image2prompt | 圖片反推提示詞 | 包含相容性修正 |
| sd-webui-prompt-all-in-one-neo | 提示詞管理與編輯 | 包含繁體中文翻譯 |
| stable-diffusion-webui-tagger-fork | 圖片標籤反推 | 包含介面與相容性修正 |
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

## Windows 安裝方式

### 需求

- Windows 10／11
- Git
- Python 3.13.12（目前 Forge Neo 上游測試版本）
- 支援 CUDA 13 的 NVIDIA 顯示卡驅動程式

### 安裝

在命令提示字元中執行：

    git clone https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git
    cd sd-webui-forge-neo-zh-TW
    webui-user.bat

首次啟動會建立 `venv`，並由 Forge Neo 安裝器自動安裝目前預設的 PyTorch／CUDA 套件及其他相依套件。它不會自動安裝 Python 本體、NVIDIA 驅動程式或 Stable Diffusion 主模型。

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

## 選用放大模型下載

本專案不直接收錄大型模型檔。若要安裝常用 ESRGAN／RealESRGAN 放大模型，請在專案根目錄執行：

    download-upscalers.bat

下載器會將下列模型放入 `models\ESRGAN`：

- `RealESRGAN_x4plus.pth`
- `RealESRGAN_x4plus_anime_6B.pth`
- `4x-AnimeSharp.pth`
- `4x-UltraSharp.pth`
- `4x_foolhardy_Remacri.pth`

已存在的模型會自動跳過；下載失敗時可重新執行下載器。完成後請重新啟動 Forge Neo，模型便會出現在放大演算法選單中。

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
- [bluelovers/stable-diffusion-webui-localization-zh_Hant](https://github.com/bluelovers/stable-diffusion-webui-localization-zh_Hant)
- 所有原始翻譯作者、插件維護者與測試使用者。

**繁體中文化與整合維護：DianNaoTou**

## 授權

Forge Neo 主程式及各插件分別依其原始授權發布。本整合版保留上游授權檔、著作權聲明及來源資訊；個別元件的使用與再散布條件，請以其原始授權為準。

---

如果這個整合版對你有幫助，歡迎提交 Issue、協助測試或改善翻譯。
