# 版本與下載說明

本整合版以繁體中文使用者為主要對象。版本號跟隨 Forge Neo 上游；本專案額外整合或修補以 `-tw.修訂號` 區分，例如 `2.29.2-tw.1`、`2.29.2-tw.2`。日常修改先進入 `neo-test`，確認後再發布至 `neo`。

## 如何選擇

| 選項 | 適用情況 | 下載 |
|---|---|---|
| 最新正式版：2.29.2-tw.1 | 一般安裝與更新 | [正式分支 ZIP](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/archive/refs/heads/neo.zip) |
| 首個穩定版：2.29.1-tw.1 | 保留更新前行為或回退比對 | [固定提交 ZIP](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/archive/fecbe6bf666a102680fcfe0626eae91a60fe0122.zip) |
| 測試版：neo-test | 協助驗證尚未正式發布的修改 | [測試分支 ZIP](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/archive/refs/heads/neo-test.zip) |

正式與測試分支 ZIP 重新下載時會取得當下分支內容；已下載的 ZIP 不會自動更新。首個穩定版固定提交 ZIP 永遠指向相同程式。兩個版本均已建立固定 Tag／GitHub Release。進入下列發布頁，展開 Assets 並選擇 Source code (zip)，即可下載固定版本：

- [2.29.2-tw.1 正式發布頁](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/releases/tag/2.29.2-tw.1)
- [2.29.1-tw.1 首個穩定版發布頁](https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW/releases/tag/2.29.1-tw.1)

固定版本 ZIP 不會跟隨分支更新；需要後续更新時，請使用正式分支或下載下一次發布。

## 本次與歷史說明

- [2.29.2-tw.1：本次更新與已知限制](docs/releases/2.29.2-tw.1.md)
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

新版啟動與介面資訊使用 `neo 2.29.2-tw.1`。舊版程式保持不變，仍顯示 `neo 2.29.1`，其整合版發布名稱為 `2.29.1-tw.1`。

## 發布紀錄規則

每次正式發布記錄上游基準、整合版新增／修正、實測結果及已知限制。正式版本使用固定 Tag／Release 保存；共同工作紀錄另記錄修改過程，不取代給使用者看的更新說明。
