# 版本與下載說明

本整合版以繁體中文使用者為主要對象。版本號跟隨 Forge Neo 上游；本專案額外整合或修補以 `-tw.修訂號` 區分，例如 `2.29.2-tw.1`、`2.29.2-tw.2`。日常修改先進入 `neo-test`，確認後再發布至 `neo`。

目前正式版：**[2.31-tw.1](docs/releases/2.31-tw.1.md)**；上一正式版為 **2.30-tw.1**。

## 如何選擇

| 選項 | 適用情況 | 下載 |
|---|---|---|
| 最新正式版：2.31-tw.1 | 一般安裝與後續 Git 更新 | [neo.zip（含 Git）](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/releases/download/2.31-tw.1/neo.zip) |
| 上一正式版：2.30-tw.1 | 回退或比對上一版本 | [歷史發布頁](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/releases/tag/2.30-tw.1) |
| 測試版：2.31-tw.1-test.4 | Intel WD14 GPU 選用實驗 | [neo-test.zip（含 Git）](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/releases/download/2.31-tw.1-test.4/neo-test.zip) |

本次正式版以使用者已測試的 **Test 3** 為來源，**不含 Test 4 的 Intel WD14 GPU 修改**。正式版 Intel WD14 維持 CPU；Intel 生圖使用既有官方 XPU profile。使用者於 2026-10-25 明確授權本次 2.31 大版本整合，此次已符合先前「下一次大版本才納入官方 XPU」的發布安排。

正式版 Release 附件 `neo.zip` 包含精簡 Git 資訊，追蹤 `origin/neo`；測試版附件 `neo-test.zip` 追蹤 `origin/neo-test`。電腦須先安裝 Git，在專案資料夾執行 `git pull` 即可更新對應分支。打包會驗證精確提交、追蹤分支與 checksum，不包含模型、虛擬環境、個人設定或輸出。

**GitHub 自動產生的 Source code ZIP 與 Code → Download ZIP 仍不含 `.git`**；舊 ZIP 不會自動轉成 Git 安裝。需要此功能請下載新版 Release 附件，解壓至新目錄；不要直接覆蓋原有個人資料。舊版本 Tag／附件均保留，不覆寫。

## 本次與歷史說明

- [2.31-tw.1：本次更新與已知限制](docs/releases/2.31-tw.1.md)
- [2.30-tw.1：歷史更新與已知限制](docs/releases/2.30-tw.1.md)
- [2.29.2-tw.1：歷史更新與已知限制](docs/releases/2.29.2-tw.1.md)
- [2.29.1-tw.1：首個穩定版保存紀錄](docs/releases/2.29.1-tw.1.md)
- [完整更新紀錄](CHANGELOG.md)

## Git 安裝與更新

全新安裝正式版：

```bat
git clone --branch neo https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git
cd sd-webui-forge-neo-zh-TW
```

已在正式分支的使用者，先關閉 WebUI，再執行：

```bat
git branch --show-current
git pull --ff-only
```

第一個指令應顯示 `neo`。若顯示 `neo-test`，更新的是測試版；不想改動現有測試目錄時，請另開資料夾安裝正式版。若 Git 提示本機修改或無法快轉，保留修改並處理差異，勿直接強制重設。

要使用固定舊版，建議在獨立資料夾下載固定 ZIP；Git 使用者可另行複製儲存庫後，切換至提交 `fecbe6bf666a102680fcfe0626eae91a60fe0122`，不要在原有使用環境直接回退相依套件。

## 版本呈現

正式版啟動與介面資訊使用 `neo 2.31-tw.1`；測試分支目前使用 `neo 2.31-tw.1-test.4`。`-test` 表示測試分支，並非新的正式發布。正式發布前移除 `-test`，並依當次更新調整上游版本及 `tw` 修訂號。舊版程式保持不變，仍顯示 `neo 2.29.1`，其整合版發布名稱為 `2.29.1-tw.1`。

## 發布紀錄規則

每次正式發布記錄上游基準、整合版新增／修正、實測結果及已知限制。正式版本使用固定 Tag／Release 保存；共同工作紀錄另記錄修改過程，不取代給使用者看的更新說明。

更新前正式分支提交 `7008f298b9c113a954bbdba92859ad82d50ab68b` 保存於 `archive/2.29.2-tw.1`，原固定 Tag／Release 不移動。發布 workflow 僅在正式 `neo` 分支更新時執行，回歸通過後依版本檔與本版文件建立固定 Tag／Release；相同版本的既有 Release 不覆寫。
