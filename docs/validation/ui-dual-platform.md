# test.5 雙平台實機驗證清單

本次工程環境沒有 RTX 5060 Ti／Arc B580；以下項目目前全部為「未測」。
自動化測試中的替身與 CPU 運算不能視為插件真實模型或 GPU 測試通過。

## 更新與啟動

在測試版目錄操作。保留現有模型連結及所有 venv，不刪除或重建環境。

```powershell
git branch --show-current
git status --short
git fetch origin neo-test
git pull --ff-only origin neo-test
```

確認分支為 `neo-test`、版本顯示 `neo 2.30-tw.2-test.5`。
如有本地修改或快轉失敗，先保留差異，不執行 reset／clean。
更新前另外備份測試版的 `config.json`、`ui-config.json`，及已使用的
`runtime/xpu-ab/official`、`runtime/xpu-ab/kitchen` 設定；不要覆蓋正式版。

- NVIDIA：使用原本的 CUDA 啟動方式與獨立環境，記錄實際 Python／Torch／驅動。
- Intel：使用原本 Intel 啟動器；若測 official profile，用
  `webui-user-intel-xpu-official.bat`（7862）；kitchen 用原有已完成 native kernel
  安裝的 `webui-user-intel-xpu-kitchen.bat`（7863）。本次不要求安裝 kitchen。
- 記錄使用的設定檔位置；XPU profile 使用自己的 config／ui-config，
  首次初始化後不會自動覆蓋為根目錄的新設定。
- 啟動沒有 traceback／插件載入錯誤。驗證 `comfy_kitchen` 不再因缺少
  `__version__` 中止（原有修補應保留）。

## 每個平台各填一次

| 項目 | RTX 5060 Ti 16GB | Arc B580 12GB | 操作與通過條件 |
|---|---|---|---|
| 啟動／環境 | 未測 | 未測 | 使用原有 venv；確認 CUDA／XPU 正確，沒有意外換核心或降級 |
| 前端資源 | 未測 | 未測 | 強制重新載入（Ctrl+F5）；F12 Network 的 script.js、style.css、localization.js、插件 main.entry.js／tagger.js 均 200，無 403／404／MIME 錯誤 |
| 視覺布局 | 未測 | 未測 | 與原 2.30 繁中版同解析度／縮放比較；模型列、正負提示詞、生成區、採樣、尺寸、HR、結果區及設定頁一致 |
| 繁中 | 未測 | 未測 | 設定語言為 zh_Hant；生成、文生圖／圖生圖、設定及插件既有翻譯正常；重啟後保留 |
| 模型與 HR 刷新 | 未測 | 未測 | 切換模型／VAE／文字編碼器，測頂部刷新與 HR 原有刷新按鈕；新選項出現，不重設其他參數 |
| 中文 Tag 插件 | 未測 | 未測 | 正負提示詞各輸入中文；用原本已設定的翻譯服務轉英文，傳送後原提示詞欄同步，生成參數含實際英文；圖生圖再測一次 |
| WD14 | 未測 | 未測 | 載入一張測試圖，選原常用模型辨識；點標籤不插入多餘反斜線，傳到文生圖／圖生圖及連圖傳送均正確 |
| WD14 新模型 | 未測 | 未測 | 選 PixAI／Camie 時才下載；各測一次辨識及快取再載入；記錄 ONNX provider／是否回退 CPU |
| 基本生圖 | 未測 | 未測 | 相同常用 SDXL 模型、固定種子、1024×1024、batch=1、20 steps，連續 3 次無空白／NaN／OOM |
| HR | 未測 | 未測 | 同組參數，1.5×、HR 10 steps、denoise 0.35，先用原常用放大器，再測原常用倍率；記錄時間與顯存 |
| 圖生圖 | 未測 | 未測 | 同張輸入、固定種子、denoise 0.4，正常出圖及跨頁傳送 |
| ADetailer-Neo | 未測 | 未測 | 啟用原常用臉部偵測模型；生成日誌確認偵測及重繪實際執行，輸出參數含 ADetailer，單有勾選框不算通過 |
| ControlNet | 未測 | 未測 | 選相容 SDXL ControlNet／預處理器，執行預覽及生圖；觀察控制效果，停用後可再生圖；模型錯誤應明確中止並能恢復 |
| 其他內建／第三方插件 | 未測 | 未測 | Image2Prompt 辨識傳送、Civitai 以原設定下載、LoRA、DeGrid、XYZ 與常用插件各完成一次原有流程 |
| 新模型支援 | 未測 | 未測 | 已有 Qwen-Image 2.1 權重時測一次；未下載者標記未測，不能宣稱已驗證 |
| 重啟與重載 | 未測 | 未測 | Reload UI 兩次、切頁、重啟，確認無重複插件面板／事件，語言、布局與設定持續正確 |

HR 的 1.5× 是可重現的起始組合，不保證所有模型／記憶體配置都可容納。
若 OOM，保留模型、解析度、放大器、RAM／VRAM 和錯誤日誌，分開記錄 UI 與
推論結果；不能把縮小參數後成功寫成原條件通過。

## 回報格式

```text
平台／GPU：
Commit／版本：
啟動器／設定檔位置：
Python／Torch／GPU 驅動：
模型／VAE／LoRA／採樣器：
解析度／種子／steps／HR：
已通過的表格項目：
失敗項目與重現步驟：
未測項目及原因：
啟動日誌／前端 Network 狀態／比較截圖：
```

翻譯服務可能需要原有 API Key 或網路；不要把 Key 貼入回報。
本次不會複製或變更使用者的翻譯憑證。
