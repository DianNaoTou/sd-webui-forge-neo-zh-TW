# Forge Neo 2.32 新功能與使用方式

本文件依上游 2.32（`a31a8ba0`）的 README 與本版程式整理翻譯，適用於繁體中文整合版 **2.32-tw.1**。上游原文保留於 [README_EN.md](../README_EN.md)。

## 新增與更新的模型功能

### Moss-Image-Edit

支援 Moss-Image 的編輯功能，需搭配指定的 [Moss-Image-Edit LoRA](https://huggingface.co/chinoll/Moss-Image-Edit)。請在「設定 → Stable Diffusion」中選擇具編輯能力模型的圖生圖／參考圖片模式；Moss-Image 的「參考圖片模式」設定保留可切換選項。

### ANIMA RGBA 透明圖片

支援 ANIMA RGBA，需搭配指定的 [LoRA 與 VAE](https://huggingface.co/localai0301/Anima-RGBA)。使用透明圖片進行圖生圖時，上游建議將 `img2img_background_color` 設為 `(255, 255, 255)`，也就是白色背景。

ANIMA RGBA 與 LayerDiffuse 是不同的透明圖片生成方式；請分別使用各自對應的模型、VAE 與設定。

### 編輯模型的參考模式

具備編輯能力的模型可在「設定 → Stable Diffusion」切換一般圖生圖與參考圖片模式。是否需要額外 LoRA、VAE 或擴充，依各模型說明而定。模型下載來源與建議參數可參考上游的 [模型下載](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)及[推論參考](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)。

## 模型選擇與比較

- 模型卡片現在能一併套用偏好的文字編碼器與 VAE。請在模型中繼資料編輯器的「偏好使用的 VAE／文字編碼器」設定；沒有附加模組的卡片仍可直接切換模型。
- 改善模型清單排序與架構中繼資料讀取，保留既有 VAE／文字編碼器短名稱與完整路徑設定。
- XYZ 圖表支援文字編碼器比較軸與 LoRA 提示詞搜尋／取代。VAE 軸選擇 None 時只移除 VAE，不移除文字編碼器。

## 新增啟動參數說明

| 參數 | 用途 |
|---|---|
| `--disable-preset` | 從下拉選單與設定中移除指定的預設模式；在參數後提供預設模式名稱。 |
| `--sort-preset` | 依字母順序排列預設模式選單。 |
| `--force-w4a8` | 將 `convrot_w4a4` 模型以 `w4a8` 格式載入；上游將此選項列為畫質改善方式。 |

例如啟動後排序預設模式：

```bat
webui-user-intel-arc.bat --sort-preset
```

NVIDIA 可將相同參數加在 `webui-user.bat` 後。其他參數與格式見[上游英文說明](../README_EN.md#by-neo)。

## GPU 與安裝流程

- 啟動時透過 PyTorch accelerator API 找出可用裝置，實際建立張量、執行運算並傳回 CPU，檢查裝置是否可運作。
- Qwen3-VL 圖片嵌入使用文字編碼器所在裝置進行視覺處理，再將輸出特徵傳回目標裝置。
- ControlNet 新安裝的 InsightFace 使用上游指定的 `2.1 --no-deps`；整合版保留 ONNX Runtime 的 CPU／GPU 分流，避免兩種套件互相覆蓋。
- Intel Arc 使用官方 XPU 生圖環境，WD14 則在獨立 OpenVINO 環境執行 GPU 辨識。辨識 GPU 初始化或推論失敗時自動退回 CPU。詳見 [Intel Arc 安裝與啟動](../INTEL_ARC_SETUP.md)。

## 語言切換

在「設定 → 使用者介面 → 語言／Localization」選擇 `zh_Hant`、套用設定並重新載入介面，即可使用繁體中文；選擇不使用翻譯並重新載入，可回到英文。主要翻譯集中於 `localizations/zh_Hant.json`；使用獨立語言選單的插件沿用其原有機制。
