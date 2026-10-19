# test.5：恢復 A1111 繁中 UI 與插件前端

## 結果與範圍

只修改 `neo-test`，版本 `neo 2.30-tw.2-test.5`；不合併 `neo`、不發布 Release。
這次完成可重現的前端資源授權修復、布局恢復及回歸保護；**尚未完成兩台 GPU
與完整瀏覽器操作驗收**。不能將本文件視為所有插件真實推論已通過。

## 基準與根因

比較正式 `neo`（`678453607fd97ae4a2a0f6d926631ce112d3161d`，2.30-tw.1）、
共同祖先 `1e1e4d6a`、測試分支起點 `446b3117` 與上游同步歷史。
正式版保留使用者原本 2.30 的 A1111 操作；test.4 不是 UI 基準。

1. **可重現的主要根因：獨立 data-dir 改變 Gradio 的預設檔案白名單。**
   `modules/cmd_args.py` 的 `--gradio-allowed-path` 預設是 `[data_path]`。
   XPU profile 設為 `runtime/xpu-ab/{official,kitchen}` 後，原始碼根目錄的
   `script.js`、`style.css`、`javascript/localization.js`、內建插件資源落在白名單外。
   extension junction 只是讓 Python 找得到插件，不能代替前端授權。
   真實 Gradio 4.40.0／FastAPI HTTP 測試重現修復前 403、修復後 200。
2. 使用者提供的 2026-10-19 17:05 HTML 仍有 Gradio 4.40.0 與原本元件，
   卻呈現英文及空的自訂樣式；這與上述根因一致。SingleFile 會移除腳本內容，
   因此不能單憑存檔推定每個資源的原始 HTTP 狀態，403 結論來自獨立重現。
3. 上游 `af59a359` 另外改變設定頁布局／自動捲動；`aa144fad` 刪除 HR 區的
   本地刷新按鈕。這些是獨立的 UI 行為差異，不是 Torch 導致整套框架更換。
4. WD14 的 `waitQuerySelector` 將已取得的 DOM 元件再次交給
   `document.querySelector`，可能丟出無效選擇器錯誤。點擊標籤還把 regex
   escaping 後的字串插入欄位，並將事件送到外層而非 textarea。

## 修復

- 啟動／Reload UI 精確授權核心及啟用插件的公開前端檔案，涵蓋
  `javascript`、`html`、`styles` 與載入的 `style.css`；保留私有設定目錄隔離。
- 前端 URL 編碼保留 Windows 路徑、中文、空白、`#`、`&` 的檔名語義；
  同時處理符號連結路徑與實體路徑。Windows junction 仍待實機驗證。
- CSS 與設定頁 JavaScript 恢復至已發布 2.30-tw.1；恢復舊設定頁按鈕標籤。
- HR 原位置恢復刷新按鈕，同時保留上游共用 checkpoint refresh callback。
- WD14 修正 DOM 查詢、標籤原文及 textarea input／blur 通知。
- 不重設 `config.json`／`ui-config.json`，不覆蓋使用者語言、插件啟用狀態、
  翻譯服務憑證或自訂 CSS。既有 `zh_Hant` 字典完整保留且新設定預設仍為繁中。
- CUDA／XPU 使用相同前端授權與 UI 程式碼；未改 Torch、requirements、
  XPU profile／依賴隔離、`comfy_kitchen.__version__` 修補、Qwen／Spectrum、
  ControlNet 生命週期及 PixAI／Camie adapter。

## 修改檔案

| 檔案 | 用途 |
|---|---|
| `modules/ui_gradio_extensions.py` | 前端檔案精確授權與 URL 編碼 |
| `webui.py` | 在所有平台的 Gradio launch 套用資源授權 |
| `style.css` | 還原發布版樣式 |
| `javascript/settings.js` | 還原原設定頁操作 |
| `modules/ui_settings.py` | 恢復原按鈕標籤 |
| `modules/ui.py` | HR 本地刷新按鈕及保留共用刷新 |
| `extensions/stable-diffusion-webui-tagger-fork/javascript/tagger.js` | WD14 DOM 與標籤輸入事件修復 |
| `modules_forge/forge_version.py` | test.5 版本識別 |
| `tests/test_ui_asset_serving.py` | 真實 Gradio HTTP、語言載入及啟用插件資源測試 |
| `tests/test_ui_freeze.py` | 發布版元件階層、樣式、刷新與 WD14 傳送註冊保護 |
| `tests/fixtures/ui_2_30_tw_1.json` | 從發布 commit 產生的 UI 基準 |
| `tests/test_tagger_frontend.js` | 執行 WD14 原始 JavaScript 的 DOM／事件測試 |
| `.github/workflows/cpu-regressions.yml` | 安裝固定 UI 測試依賴及執行 Node 測試 |
| `docs/UI_FREEZE.md` | 長期 UI 保護與上游整合規範 |
| `docs/validation/ui-dual-platform.md` | RTX 5060 Ti／B580 實機驗證清單 |
| `docs/reviews/2026-10-19-ui-freeze-test5.md` | 本次工程紀錄 |
| `CHANGELOG.md`、`README.md` | 版本紀錄與規範入口 |

## 本輪驗證

- Python 3.12、真實 Torch 2.14.1+cpu：完整 **116 項 unittest 通過，0 skipped**。
  包含現有 ControlNet 生命週期、VAE／Flash、WD14 adapter、XPU profile、
  安裝政策、翻譯與新增 UI 測試。
- 真實 Gradio 4.40.0、FastAPI 0.127.1、HTTPX 0.28.1 檔案路由：
  核心／提示詞插件／WD14 腳本與 CSS 在獨立 data-dir 下恢復可讀；
  已注入資源均在白名單，翻譯合併與注入順序正確，非公開設定仍拒絕讀取。
- `python -O` 另執行 VAE／Flash、ControlNet、XPU profile、WD14 共 **46 項通過**。
- **4 項 Node JavaScript 測試通過**：已存在元件、延遲掛載、逾時、標籤事件。
- Python compileall 通過；原有第三方 `translators/server.py` 有一條
  invalid escape sequence SyntaxWarning，未改寫該第三方模組。
- 差異檢查以 `core.whitespace=cr-at-eol` 保留既有 CRLF 檔案格式。

WD14 adapter 測試使用模擬 ONNX session／下載器，不能把測試日誌中的
「已載入」當成這輪下載新權重或真實模型辨識。HTTP 測試替換 Forge GPU 服務，
也不代表完整 WebUI／ADetailer Python 載入或瀏覽器 Vue 掛載已驗收。

## 插件盤點及尚未驗證

| 功能 | 已檢查 | 尚待實機 |
|---|---|---|
| Prompt All-in-One Neo | 原前端路徑、核心腳本順序、真實 HTTP 可讀；未改 API／憑證流程 | Vue 面板掛載、真實中文翻譯服務、正負提示詞與圖生圖同步 |
| WD14 | 前端掛載程式、標籤事件、三組原始跨頁 binding、既有 adapter 回歸 | 真實 ONNX 辨識／provider、瀏覽器傳送及圖像同步 |
| ADetailer-Neo | 無版本間程式差異；保留原 UI／生命週期；既有 ControlNet 巢狀隔離測試 | Python 真實完整載入、偵測權重與實際二次重繪 |
| ControlNet／IPAdapter | 保留 test.2 修復、完整既有 CPU 生命週期回歸 | 預處理預覽、匹配模型控制效果、HR 與重試 |
| Image2Prompt／Civitai downloader／其他內建插件 | 啟用插件公開資源收集、compileall、既有相關測試 | 真實模型、下載授權、外部網路及完整 UI 操作 |
| NVIDIA／Intel | 前端無平台分歧；CPU 回歸及 XPU 隔離契約 | Windows 啟動、GPU 生圖、HR、顯存／RAM、效能及截圖比對 |

後續風險：使用者 profile 已保存的語言／停用插件／自訂 CSS 仍可能影響外觀；
未自動重設。未知第三方插件若引用公開目錄以外的檔案，需要新增精確授權與測試。
沒有宣稱 GPU 性能改善或所有插件完整可用。

## 還原

以一般 `git revert` 還原本修復提交，不使用 reset／強制推送；
因未更動模型、設定或 venv，還原不需要重建環境。
