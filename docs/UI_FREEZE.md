# A1111 介面保護規範（UI Freeze）

## 基準與範圍

基準是已發布、使用者確認過操作方式的 `neo 2.30-tw.1`：
`678453607fd97ae4a2a0f6d926631ce112d3161d`。不能把異常的 test.4 當成新基準。
`neo` 與 `neo-test` 共用此規範；目前修復僅提交 `neo-test`。

保護文生圖／圖生圖的提示詞、生成按鈕、尺寸、採樣器、步數、HR、模型及
VAE／文字編碼器選擇、設定頁導航與既有插件操作流程。保留 `zh_Hant` 預設、
完整翻譯資源、既有元件 ID、插件 callback 與提示詞傳送掛載點。

## 更新規則

1. 上游同步先分類為後端、依賴、UI、插件掛載；UI 差異逐項審查，不整包覆蓋。
2. CUDA／XPU 共用同一套 UI 程式碼。平台判斷留在推論、依賴與硬體能力層；
   不因 PyTorch、獨立 venv、`--data-dir` 或 GPU 改變元件位置與操作方式。
3. 新功能優先放進原有頁籤／群組；新增可選項不得替換原本控制項、移動提示詞
   或生成區。硬體不支援時使用既有控制項的說明／停用狀態。
4. 保留插件前端、後端與跨頁傳送；不能以停用插件、重設使用者 config、
   刪除翻譯、降級 Torch、移除新模型或 XPU 修補作為修復方式。
5. 前端資源授權由 `modules/ui_gradio_extensions.py::ui_asset_paths` 集中處理，
   於啟動／UI 重載重新產生；只授權已啟用插件的公開前端檔案。
   不為修復 CSS／JavaScript 而開放整個原始碼或插件根目錄。
6. 維持 `config.json`、`ui-config.json` 的使用者設定及各平台獨立狀態目錄。
   `localization=zh_Hant` 是新設定的預設值，不強制覆蓋使用者已選語言。
7. 設計更動必須先取得使用者明確同意。不得僅為讓 CI 變綠而更新基準快照。
   核准後在同一提交記錄原因、前後差異及實機結果，再更新保護範圍。

## 自動檢查

```bash
python -m unittest discover -s tests -v
node --test tests/test_tagger_frontend.js
```

`tests/fixtures/ui_2_30_tw_1.json` 取自上述發布 commit：

- 保護 19 個重要元件宣告及其容器階層、標籤、預設值（涵蓋動態文生圖／圖生圖 ID）。
- 鎖定發布版 CSS 及設定頁 JavaScript，防止上游直接改變既有布局。
- 執行真實 Gradio 4.40.0 的 HTTP 檔案路由：修復前獨立 data-dir 的 403，
  修復後 CSS、翻譯腳本及插件 JavaScript 的 200、內容與 MIME 類型。
- 驗證 `zh_Hant` 實際合併結果、翻譯先於腳本注入、資源白名單完整性、
  停用插件排除、特殊檔名與符號連結，以及非公開設定檔仍不可讀取。
- 執行 WD14 JavaScript 的既有／延遲元件掛載、逾時及標籤輸入事件；
  執行原始 Python 跨頁傳送註冊，檢查提示詞與圖片來源。

CI 對 `neo-test` push／PR 執行上述測試，以及既有 ControlNet、WD14 模型、
VAE／Flash、XPU profile 與安裝政策回歸。CI 安裝真實 CPU Torch 與固定 Gradio
依賴；GPU 推論與瀏覽器視覺比對不可用這些測試代替。

## 上游新功能整合範例

- 保留上游共用 checkpoint refresh callback；在原 HR 區恢復自己的刷新按鈕，
  同時保留共用刷新功能。
- 保留 Spectrum、新 WD14 模型 adapter、ControlNet 生命週期、新模型與 XPU
  依賴成果；它們不能連帶覆蓋設定頁 CSS 或切換 UI 框架。
- 新增第三方插件若引用 `javascript`、`html`、`styles` 以外的公開資源，
  先新增精確授權與 HTTP 測試，再啟用該功能，不直接開放其設定或儲存目錄。

## 人工驗收

正式合併前依 [雙平台實機清單](validation/ui-dual-platform.md) 逐項記錄。
必須分別標示「通過／失敗／未測」，保留啟動日誌、生成參數及截圖比較。
CPU 測試通過不代表 NVIDIA 或 Intel 實機驗收通過。
