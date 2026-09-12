# 更新紀錄

本專案的重要變更會記錄在此處。

## 尚未發布

### 新增

- 以 Forge Neo 最新 `neo` 分支為基礎建立 DianNaoTou 台灣繁體中文整合版。
- 加入完整 `zh_Hant` 語言檔，共 9,142 條介面翻譯。
- 全新安裝預設使用台灣繁體中文介面。
- 整合 ADetailer-Neo。
- 整合 forge-neo-image2prompt，包含本機相容性修正。
- 整合 sd-webui-prompt-all-in-one-neo，包含繁體中文翻譯。
- 整合 stable-diffusion-webui-tagger-fork，包含介面與相容性修正。
- 增加繁體中文 README，並保留上游英文原文。

### 安全與容量

- 排除 Stable Diffusion、LoRA、VAE、Embedding、YOLO 等模型檔案。
- 排除使用者生成圖片、影片、`outputs`、`venv`、快取與備份檔。
- 插件所需模型仍依各插件原有機制在需要時下載。

### 執行環境

- 跟隨 Forge Neo 最新上游需求。
- 目前上游測試環境為 Python 3.13.12。
- PyTorch／CUDA 套件由 Forge Neo 安裝器於首次啟動時自動安裝。
