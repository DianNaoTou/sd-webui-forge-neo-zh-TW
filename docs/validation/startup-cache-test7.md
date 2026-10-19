# Test 7：啟動版本標記與圖片快取

版本：`neo 2.30-tw.2-test.7`。僅更新 `neo-test`，不變更正式 `neo` 或建立正式 Release。

## 問題與修正

1. 官方 XPU 在上游檢查之前建立 config.json，但全新設定沒有 `VERSION_UID`。
   新設定現在寫入共用常數 `PY313`；從原根目錄複製的舊設定保留原標記，仍需確認。
   既有官方設定顯示版本提醒後，按 Enter 才先備份原檔，再只更新標記。
   中斷確認不寫入，非官方 profile 不回寫，NVIDIA 原設定不受此確認流程修改。
   警告改述為設定格式問題，不再直接要求重新安裝套件或刪除模型。
2. Gradio 4.40 在 Windows 遇到同名目標時，rename 可能失敗並改走背景 shutil.move。
   背景複製可能覆寫正在讀取的圖片。現在相同內容不覆寫；需要複製時先寫同目錄
   暫存檔、關閉 handle，再原子替換目標。原 PNG bytes／metadata 保留。
   PIL 預覽暫存檔也明確關閉後才返回路徑。錯誤不會被一律吞掉或無限重試。

使用者 21:09 的日誌確實是 Gradio FileResponse 讀取快取 PNG 被 PermissionError 拒絕，
不是缺套件，也不是 Gradio allowlist 403。可重現的上傳覆寫競爭已修正；
無法只憑該日誌證明原機一定是此競爭造成，防毒／ACL／其他占用仍須實測區分。

## 隔離與不變範圍

- 官方 XPU 的 venv、設定、快取仍在原位置，快取為 `runtime/xpu-ab/official/cache/gradio`。
- NVIDIA 保留原本自己的環境與快取；沒有合併快取、搬移資料或強制修改 temp_dir。
- 共用的是檔案處理邏輯，兩平台目錄仍分開。測試確認兩個 cache 的 HTTP 服務互不放行。
- 不更動 UI／繁中／插件掛載、Torch／CUDA／XPU 依賴、採樣、VAE 設定或模型支援。

## 更新與回退

關閉 WebUI 後，在 Windows CMD 執行：

```bat
cd /d D:\stable-diffusion\neo-test
git branch --show-current
git status --short
git pull --ff-only origin neo-test
webui-user-intel-arc.bat
```

確認分支 neo-test；有本地修改或快轉失敗時保留差異，不做 reset／clean。
不刪 venv；啟動版本應顯示 test.7。第一次出現設定確認就按 Enter，之後重啟應不再出現。
原設定備份位於 `runtime/xpu-ab/official/config-before-version-*.json`。
回復設定時先關閉 WebUI，備份目前 config，再將選定備份複製回 config.json。
程式回退可使用更新前提交 `f181b60b07f44fc793f1c482622296ccf96ad654` 建立獨立副本；
勿刪除模型／原有 venv 或直接覆蓋正式分支。

## 實機驗證（本輪 GPU 尚未驗證）

RTX 5060 Ti 與 Arc B580 各自驗證：

1. 啟動與重啟；確認 CUDA／XPU 與既有 venv 正確，NVIDIA 不出現 Intel 設定回寫。
2. 圖像資訊上傳原 PNG，連續上傳同一張 5 次並切頁，圖片與資訊均顯示，沒有 PermissionError。
3. 傳送到文生圖與圖生圖，核對正負提示詞、尺寸、步數、種子；圖生圖輸入圖片正確。
4. 完成基本生圖與 HR 各兩次，再上傳／傳送；檢查預覽、存檔與系統反應。
5. B580 保持「對 VAE 啟用（始終分塊）」勾選、UNet 始終卸載不勾；記錄是否還有 OOM。
6. 若仍有 PermissionError，保留完整 traceback、發生操作與快取路徑；不要以管理員權限或重裝掩蓋。

## 自動化驗證

- `python -m unittest discover -s tests -v`：真實 CPU PyTorch 與固定 Gradio/FastAPI。
- `tests/test_upload_cache.py`：HTTP 上傳／下載、相同內容不覆寫、分離 cache、不放寬 403、關檔後發布、持續權限拒絕仍報錯。
- `tests/test_settings_version.py`、`tests/test_xpu_profiles.py`：新設定無誤報、舊設定需確認、備份保留原 bytes、只改標記、取消不寫入、NVIDIA 不回寫。
- Windows CI 用 CreateFileW 讀取共享鎖，實際拒絕背景覆寫；Linux 用替身模擬 Windows 失敗條件。皆不代表 GPU 生圖驗收。

本輪本機結果：139 項 Python 回歸全部通過（零跳過），24 項最佳化模式測試、4 項 Node 前端測試及編譯／diff 檢查通過。Windows CI 與實機驗證須另依實際執行結果判定。
