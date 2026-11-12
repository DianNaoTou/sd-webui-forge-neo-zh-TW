# Prompt All-in-One 更新整合（2026-10-26）

本次僅更新 `neo-test`，維持現有測試版本編號，不建立新 Release／Tag。

## 來源

- 使用者指定原版：https://github.com/Physton/sd-webui-prompt-all-in-one
- 最新 main：`e66d417c37097e6383f81b78d4fd49fdb0081f10`，2026-10-13。
- 現有 Neo 相容來源：https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo
- Neo 最新 v0.3.5：`cecd70ed68c4c48dbd648d3acb16162623fd03fa`。
- 共同基底：`596a931725209a9ee271e6a2bd6d95a1cdb33778`。
- 本次專案基底：`42832775e9347d4d6a466cb10f59626f03194b85`。

## 合併決策

保留插件目錄 `extensions/sd-webui-prompt-all-in-one-neo` 與 Neo 相容修改。
原版 token counter 在沒有 sd_hijack 時回傳零；保留 Neo 的模型層 token counter API，避免功能退化。
前端合併兩側原始碼後重新建置，保留品質預設、visualBreakSeparator、拖曳防護及 LoRA 尾綴精確匹配修正。
外部輸入同步不強制格式化；使用者手動編輯標籤與主動格式化仍套用設定。
上游分句原本重新建立標籤時會丟失 disabled 狀態；改用 `_restoreTag` 保留標籤屬性。
空行保留選項補入啟動時的設定讀取清單。新解析選項均預設關閉；新增樣式需自行勾選。
既有 zh_TW 預設、繁中群組標籤及 storage／歷史／收藏內容保持。
未更新 Torch、啟動器、模型或核心 UI 佈局，也未覆蓋其他插件。

## 驗證

- Vite production build、Less CSS build 通過（沿用上游既有 eval 警告）。
- Node 語法檢查通過。
- 原版 parser 測試：211 項通過，使用既有快照，未重新產生快照。
- Neo 提示詞同步及整合測試：12 項通過，涵蓋 LoRA 觸發詞移除、多 LoRA、黑名單、翻譯完成、精確字串保留、停用標籤分句與空行。
- 專案 UI Freeze：5 項通過。
- Gradio HTTP 資源測試：本機缺少固定 Gradio 依賴，4 項跳過；由既有 CI 再驗證。
- 完整瀏覽器、CUDA／XPU 生圖及 Forge Couple 搭配：未測。

## 重現

```bash
cd extensions/sd-webui-prompt-all-in-one-neo/src
npm ci --ignore-scripts
node --test ../tests/prompt-sync.test.cjs test/*.node-test.js
npm run build
node build-css.js all
```

建置鎖定使用 npm package-lock.json；pnpm-lock.yaml 保留上游原檔。
使用者更新 neo-test 後應完整重啟 WebUI，重新整理瀏覽器，確認 LoRA 加入／移除、提示詞分行與 Forge Couple 操作。
