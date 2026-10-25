# 更新紀錄

本專案的重要變更會記錄在此處。

## 2.31-tw.1-test.1（2026-10-25）

- 完整合併上游 2.31 至 `0c388f8b`，新增 MOSS-Image、Qwen3.5 文字編碼器、SLG、融合 QKV LoRA 與繁中 JSON 構圖編輯器；保留 Test 9 全部功能。
- 同步 Alpha 門檻、X/Y/Z VAE／成本排序、CPU 量化判斷、混合精度與 GGUF 修補。修正字串前綴篩選、None VAE 檔名、清理失敗阻斷還原、數字 LoRA 強度，Spectrum 在 `-O` 的未賦值崩潰，以及 JSON 載入／跳脫／可選欄位保留問題。
- 新 CUDA 預設 Torch 2.14.0／vision 0.29.0，Intel 維持官方 2.14.1+xpu／0.29.1+xpu；Kitchen 升至 0.2.40。保留 Gradio／Pillow／HEIF 相容組合，新增 MarkupSafe 2.1.5 約束。Python bootstrap 與 Nunchaku 已驗證安裝矩陣暫維持。
- 保留 A1111 UI Freeze、XPU 隔離與注意力修補、既有 ONNX 選用參數、插件、設定／模型／輸出和 ZIP Git 更新機制。僅更新測試分支，正式分支不變。
- 新增真實 CPU 小型 MOSS／Qwen 數值、SLG、RGBA、VAE 還原及 JSON 前端事件測試，擴充 Linux／Windows CI。完整 GPU 模型與品質仍待實機驗證。[本版說明](docs/releases/2.31-tw.1-test.1.md)。

## 2.30-tw.2-test.9（2026-10-23）

- 新增 LayerDiffuse 繁體中文插件，功能預設關閉；以官方原碼與 Forge Neo Fix 3 為來源，排除第三方 Git 資訊與 Python 快取，不新增套件安裝器。
- SDXL LoRA 轉接目前 WeightAdapter，檢查修補鍵與套用完整性；合成條件在插件採樣 modifier 中處理並對齊 batch，不改全域 is_inpaint 或核心推論程式。
- 透明 VAE 依工作、模型家族及編解碼用途分開快取；沿用核心 finally 清理介面還原插件掛接。增加模型／圖片／批次驗證，失敗停止採樣。
- 測試階段禁止與 ADetailer 同時啟用；SD1.5 不支援其他 Attention 掛接，權重固定為 1。保留英文 API／圖片資訊值，介面選項與提示繁中化。
- 14 種按需權重固定 Hugging Face revision 與 SHA-256；既有檔案驗證失敗時保留原檔並停止，不覆寫使用者資料。
- 本機 16 項無 GPU 回歸通過；全套 CI 與 CUDA／XPU、Hires、Illustrious、長時間連續生成另行驗收，不宣稱已通過 GPU 測試。[說明與清單](docs/releases/2.30-tw.2-test.9.md)。

## 2.30-tw.2-test.8（2026-10-23）

- 新增 Release 附件 `neo-test.zip`：固定於通過 CI 的提交，包含單分支、depth 1、無 tags 的自包含 Git 資訊；解壓後直接 `git pull` 更新 `neo-test`，不要求初始化腳本。
- 封裝預設只允許快轉、停用自動 stash，並拒絕覆寫與遠端新增檔案碰撞的 ignored 檔案；本地修改衝突時停止。不執行強制 reset／clean、模型或環境備份。
- 新增真實 Git 更新測試及追蹤資料檢查；排除個人 `webui.settings.bat`。既有受追蹤啟動檔仍受 Git 管理，不以 ignore／skip-worktree 隱藏差異。
- Linux／Windows 回歸通過後建立固定測試 Tag／Prerelease，附 SHA-256；不改正式 `neo`、正式發布機制或 Latest 正式版。GitHub archive／Source code ZIP 仍不含 `.git`。
- 不改 Test 7 的推論、CUDA／XPU 版本、模型下載或環境路徑。舊 Intel 正式版遷移限制與尚待 GPU 實機驗證項目維持；[完整說明](docs/validation/git-zip-test8.md)。

## 2.30-tw.2-test.7（2026-10-19）

- Intel 全新設定寫入與啟動器一致的 `PY313` 版本標記；改正「設定不符」被誤述為需要重裝的警告。既有官方設定仍先顯示警告，使用者按 Enter 後備份原 JSON，只更新版本標記，後續不重複提示；不自動認定舊設定已遷移，也不改 NVIDIA 既有設定檔。
- 修正可重現的 Gradio Windows 同名上傳 fallback 覆寫競爭：相同內容的既有快取不覆寫，其他複製寫入同目錄暫存檔，關閉寫入 handle 後原子發布。PIL 預覽檔也明確關檔後才返回。真正的權限拒絕仍回報，不放寬檔案存取規則。
- CUDA／XPU 仍使用各自快取；不改 UI、推論、VAE 選項或依賴版本。保留 test.6 已提交的提示詞空白欄位修正；使用者曾回報重啟後原操作已恢復，不將兩種症狀直接視為同一根因。
- 新增實際 Gradio HTTP 重複上傳／讀圖與快取隔離測試、原檔備份／版本確認測試；Windows CI 使用原生檔案共享鎖驗證，Linux 模擬 Windows rename 行為。使用者原機的 PermissionError 根因未完全定案，仍須實機複驗。
- [Test 7 更新與驗證](docs/validation/startup-cache-test7.md)。只更新 `neo-test`，不合併正式分支或建立正式 Release。

## 圖像資訊提示詞傳送修復（2026-10-19，版本維持 test.6）

- 修正空白 Gradio 欄位初始值為 None 時，傳送參數錯用 NoneType 轉換而靜默略過提示詞；文生圖／圖生圖共用的解析路徑均套用修正。
- 新增實際 Gradio 元件回歸：讀取帶參數的 PNG、正負提示詞與生成參數回填、圖生圖圖片／尺寸傳送及分頁事件綁定；保留既有欄位型別轉換與缺少欄位略過行為。
- 僅修改 `neo-test`；Windows 完整介面操作仍待使用者確認，不涉及推論或套件版本。

## Intel Arc 單一啟動入口（2026-10-19，版本維持 test.6）

- 官方 XPU 流程直接放入 `webui-user-intel-arc.bat`，移除多餘的 `webui-user-intel-xpu-official.bat`。
- 沿用既有官方環境、設定與參數處理；不改依賴或推論行為。取代下方先前的雙入口安排。

## Intel Arc 啟動入口統一（2026-10-19，版本維持 test.6）

- 原 Intel Arc 啟動檔直接轉交新版官方 XPU，保留額外參數與結束狀態；不再建立或使用 Torch 2.14.0／`venv-intel-arc`。
- 沿用既有官方 venv、設定與 7862，不刪使用者舊資料。記錄 B580 開啟 VAE 分塊後明顯改善的使用者回報。
- 僅在 `neo-test`；正式版仍待下一次大版本整合。

## Kitchen 社群實驗撤回（2026-10-19，版本維持 test.6）

- 依使用者決定暫停採用 Kitchen 社群 XPU 方案，移除其專用啟動檔、建置腳本、fork／原生核心安裝與驗收分支；只保留新的官方 XPU profile。
- 官方環境路徑、版本與 UI／插件／注意力模式不變；保留主程式必要的 `comfy-kitchen` 官方依賴。不刪使用者本機舊環境、輸出或快取。
- 完整 122 項 Python 回歸通過。歷史實驗文件保留並標記撤回；目前只在 `neo-test`，下一次大版本更新才納入正式版。

## 維護／發布政策備註（2026-10-19，程式版本不變）

- Intel 後續以新的官方 XPU profile 為主線，舊 Torch 2.14.0／`venv-intel-arc` 路徑列為淘汰；Kitchen 原列可選實驗，已由上方同日撤回決策取代。
- 新官方環境目前僅留在 `neo-test`，下一次大版本更新才納入正式版，不能以目前的一般修訂版提前合併。本次僅記錄政策，不刪環境、不改啟動器、不修改 `neo` 或發布 Release。
- [決策與當前實測狀態](docs/decisions/2026-10-19-intel-official-baseline.md)。

## 2.30-tw.2-test.6（2026-10-19）

- Intel XPU 實驗預設改為原生 PyTorch SDPA 自選後端，取消本地強制 Math／query 分塊，以測試新版核心是否仍需要舊 workaround；不宣稱 NaN／OOM 已修復或速度已改善。
- 新增 `--xpu-sdpa native|math|sliced`：`math` 對照加入分塊前的 Math 狀態，`sliced` 回到 test.5 的原保護路徑；不自動換模式掩蓋失敗，日誌與 profile 報告記錄模式。
- 不改 CUDA、模型按需載入／卸載、Torch／XPU 依賴、UI／翻譯／插件。125 項 Python 回歸、26 項最佳化模式與 4 項 JavaScript 測試通過；實機生圖／HR 尚待使用者測試。
- [三模式實機對照與回復命令](docs/validation/xpu-sdpa-test6.md)／[工程紀錄](docs/reviews/2026-10-19-xpu-sdpa-test6.md)。

## 2.30-tw.2-test.5（2026-10-19）

- 修正獨立 `--data-dir` 導致核心 CSS／JavaScript 與插件前端被 Gradio 拒絕存取；精確授權公開資源，CUDA／XPU 共用相同行為。
- 恢復 2.30-tw.1 的設定頁樣式與操作、HR 本地模型刷新按鈕，保留上游共用刷新與所有後端整合。
- 修正 WD14 DOM 選擇器及標籤輸入事件，不再把 regex 跳脫字元寫入提示詞。
- 新增 UI Freeze 規範、發布版元件基準、真實 Gradio HTTP 與 Node 事件測試；116 項 Python 測試與 4 項 JavaScript 測試通過。
- 沒有重設使用者設定、變更 Torch／XPU 依賴或發布 Release；雙平台生圖、HR 與完整插件實機驗收待測。[工程紀錄](docs/reviews/2026-10-19-ui-freeze-test5.md)／[實機清單](docs/validation/ui-dual-platform.md)。

## 2.30-tw.2-test.4（2026-10-19）

- 原 WD14 Tagger 介面新增 PixAI Tagger v0.9（官方連結之 DeepGHS ONNX）與 Camie Tagger v2（官方 ONNX），使用專用前處理與標籤解析，不替換舊模型載入器。
- 選取新模型、上傳圖片與啟動 WebUI 均不下載；首次按辨識／批次辨識或發出 API 辨識請求才下載。固定來源 revision，沿用 HF 快取，先查本機；已快取時可離線載入。
- 新模型下載／載入失敗顯示繁中錯誤並可重試，不影響其他模型；保留原標籤過濾、排序、批次與提示詞／圖片傳送流程。
- 不新增執行相依套件、不調整 PyTorch／TorchVision／ONNX Runtime 安裝或 CUDA／XPU 啟動檔；Intel Arc 沿用 CPU Tagger 後端。
- 新增 16 項測試，完整 107 項回歸通過；兩個新模型已完成實際權重 CPU 辨識、重複辨識及禁止網路後卸載重載。尚待 Windows／RTX 5060 Ti／B580 實機驗證。[完整說明](docs/reviews/2026-10-19-wd14-test4.md)。

## 2.30-tw.2-test.3（2026-10-18）

- 新增官方 XPU 與 Kitchen XPU 實驗組獨立啟動檔，兩組固定 torch 2.14.1+xpu／torchvision 0.29.1+xpu，以 Kitchen 為主要比較變數；保留原 Intel 2.14.0 啟動檔及 NVIDIA 環境。
- 分開 venv、設定、UI 設定、styles、輸出、HF／Torch／pip／暫存快取、連接埠及 Torch 安裝約束；只共用程式碼與模型路徑。擴充安裝設定階段序列化，避免 A/B 同時安裝衝突。
- 實驗組固定來源版本，提供原生建置入口與本機 Wheel／SHA256 安裝；檢查 Torch ABI、BMG package／AOT target、B580 裝置、XPU 能力與強制原生 INT8 smoke test，缺少配套時停止，不靜默回退冒充加速。
- Kitchen 修改版以獨立 requirements／安裝約束維持 0.2.37，官方組維持 0.2.38；不改根 requirements 或降級其他環境。
- 新增 12 項隔離／錯誤防護測試；[操作、建置條件與未驗證限制](docs/reviews/2026-10-18-xpu-ab-test3.md)。目前無公開配套 Windows 原生 Wheel，B 組須建置後實測；未宣稱兩組已通過 Windows／GPU 推論。

## 2.30-tw.2-test.2（2026-10-18）

- ControlNet 未選模型、模型不存在／無法辨識、已知 IPAdapter／一般 ControlNet 搭配錯誤時，明確提示並中止本次生成，避免悄悄退回無控制出圖。
- 初始化與採樣掛載失敗後清除單元參數，不再接續觸發 `KeyError: 0`；在 `process_images` 的 finally 執行可重複呼叫的請求清理，恢復掛載前 UNet。
- 修正一般 ControlNet 快取只保留權重、複本另建顯存包裝器的生命週期；快取與複本共用受管理包裝器，清理時保留記憶體管理器的追蹤，以便按需求卸載，不隱藏洩漏警告。
- 新增 CPU 回歸測試：錯配提前攔截、失敗中止與重試、多單元／HR 回呼、ADetailer 停用 ControlNet 的巢狀請求隔離、快取／複本生命週期與 dtype 分離。
- [完整修補說明與重測項目](docs/reviews/2026-10-18-controlnet-test2.md)。本輪尚無 GPU／Windows 修補後實測，不能宣稱記憶體洩漏已完全消除；正式 neo 維持不動。

## 2.30-tw.2-test.1（2026-10-18）

- 同步上游 `ce6848b8..f7f23149`，依實際 Diff 保留已整合 VAE 與我方 Flash 遮罩保護。
- Spectrum 重寫：分批／條件獨立預測、擬合快取；補上 HR 步數為 0 的回退、純量條件防護及預測串流讀取前等待。
- 修正權重錯誤分類、API 腳本預設參數重用，統一主模型／Refiner／HR／XYZ 模型清單刷新。
- 更新 Callback 呼叫者查找、設定頁排版、輸入框高度與 comfy-kitchen 0.2.38；保留本地依賴約束及繁中翻譯。
- Nunchaku 維持選用，移除自動降級 PyTorch；未確認相容 Wheel 的組合略過安裝，明確指定的 Wheel 以 `--no-deps` 安裝。
- [完整審查、衝突與驗證限制](docs/reviews/2026-10-18-upstream-sync.md)。此版未經 GPU 推論實測；正式版維持 2.30-tw.1。

## 2.30-tw.1（2026-10-13）

將已測試的 2.30-tw.1-test 升為正式版；完整新增功能、Bug 修正、驗證與已知限制見 [本次發布說明](docs/releases/2.30-tw.1.md)。

- 新增 Qwen-Image-2.1、DeGrid、正向／反向條件批次計算，以及 VAE／文字編碼器檔名顯示設定。
- 正式納入 NVIDIA CPU-only PyTorch 自動修復與 Intel XPU 後端防護；插件共用安裝流程約束 torch／torchvision 版本。
- 修正 DeGrid 寫死 CUDA、Qwen 2.1 分塊 VAE 四通道配置、Flash 遮罩在 Python -O 下被忽略，以及批次提示詞生成資訊。
- 補齊 2.30 新增繁中介面，修正提示詞延遲說明翻譯。
- 保留既有 XPU 查詢分塊與相依政策；未納入 ede60e1a causal Flash 優化，也未追加尚未測試的上游更新。
- B580 仍有 OUT_OF_DEVICE_MEMORY／DEVICE_LOST，本次未修復；依維護者決定列為已知限制後發布。
- 保留 2.29.2-tw.1／2.29.1-tw.1 固定 Tag 與 Release，另封存更新前正式分支。新增正式合併後的回歸與固定 Tag／Release 發布流程；neo-test 保持測試版標示。

## 2.30-tw.1-test（2026-10-11）

- 候選 PR：納入上游 `58700b99` 的 2D 分塊 VAE 解碼通道數修補，支援 Qwen 2.1 四通道輸出；新增數值回歸測試。
- 候選 PR：Flash Attention 有明確遮罩時保留 SDPA 分流，改用不受 Python `-O` 影響的保護；上游 `ede60e1a` causal 優化因套件版本／非方形對齊／任意遮罩風險保留待審。詳見 [技術審查](docs/reviews/2026-10-11-vae-flash.md)。
- `neo-test` 併入上游 Forge Neo 2.30（`ce6848b8`）：新增 Qwen-Image-2.1 支援、SDPA 注意力調整、DeGrid 腳本與多項小修；版本資訊改為 `neo 2.30-tw.1-test`。
- 保留本整合版修改：Intel XPU Math SDPA、Gradio 4.40／Pillow 10.4.0 相依組合、None／Ignore 提示詞權重修正與 SD 放大模型選單。
- 強化 PyTorch 後端防護（延續 Issue #4／`7e1b13e5`、`2007f401`）：Intel Arc 流程同樣偵測非 XPU 版 PyTorch 並自動重裝；重裝後仍非預期後端即停止啟動，不再改用 CPU；之後的插件安裝以已安裝的 torch／torchvision 版本為約束，避免被替換成 CPU 版。
- 修正上游新增的 DeGrid 後處理寫死 CUDA（`cf4f2e73`）：改用 Forge 偵測到的裝置（CUDA／XPU／CPU），該裝置失敗時自動改用 CPU，修正 Intel Arc 在「附加功能」處理圖片／影片時出錯；並新增測試，禁止主程式寫死 `.cuda()`。
- 補齊 2.30 新增介面的繁體中文翻譯（`85cdb821`）：DeGrid、Qwen 2.1 參考圖片、批次套用正向/反向調整、VAE／文字編碼器下拉選單顯示方式等。
- 修正「提示詞編輯傳送至 Gradio 前的延遲」說明翻譯未生效（`0feba725`），並新增語系檔格式、重複鍵與設定說明括號的檢查測試。
- 新增 CPU 環境可執行的回歸測試；尚待 RTX 與 Intel Arc 實機驗證，正式版仍為 `neo 2.29.2-tw.1`。

## 2.29.2-tw.1-test（2026-10-05）

- `neo-test` 的啟動與介面版本資訊改為 `neo 2.29.2-tw.1-test`，以 `-test` 後綴區分測試版。
- 本次僅調整版本標示與說明，生成行為及相依套件不變；正式版仍為 `neo 2.29.2-tw.1`。
- 正式發布前應移除 `-test`，並依當次更新調整上游版本及 `tw` 修訂號。

## 2.29.2-tw.1（2026-10-04）

本次正式分支更新由 2.29.1 升至 2.29.2-tw.1；上游版本為 2.29.2。更新內容、實測及已知限制見 [本次發布說明](docs/releases/2.29.2-tw.1.md)。

- 納入下方 2.29.2 測試階段的上游更新與相依修正。
- 修正 None／Ignore 提示詞權重模式回傳空值。
- XPU 查詢分塊後，使用者確認 B580 高解析度修復可完成，但速度較慢；後續曾遇 OUT_OF_RESOURCES／DEVICE_LOST，完整重啟後未再復現，尚未確認根因與連續生成穩定性。
- 使用者確認 RTX 5060 Ti 啟動與生成正常；XPU 修補後 NVIDIA 速度尚待重測。
- 新版版本資訊採用 2.29.2-tw.1，補上繁體中文版本與下載說明。
- 舊正式版 fecbe6b 以 2.29.1-tw.1 首個穩定版留存，不更動原程式。

## 2.29.1-tw.1 首個穩定版（2026-10-04 補記）

保存更新前正式提交 `fecbe6bf666a102680fcfe0626eae91a60fe0122`，程式仍顯示 2.29.1。整合版發布名稱與固定下載見 [保存紀錄](docs/releases/2.29.1-tw.1.md)。

## 歷史整合與測試紀錄

以下保留各項修改當時的狀態；正式發布時的最新驗證結果以上方發布說明為準。

### 2.29.2 測試版（2026-10-04）

- 上游同步至 [`d70373e`](https://github.com/Haoming02/sd-webui-forge-classic/commit/d70373ebcf1a96d210b78cd6f77196459e783e2a)，版本由 2.29.1 更新至 2.29.2。
- 修正 Klein、Krea2、Qwen Image 與 Z-Image 文字編碼器的單段／多段提示詞分詞處理，調整 SD／SDXL 分詞設定。
- Never OOM 的 UNet 強制卸載改為獨立控制；切換設定時只卸載相關 UNet，不再切換全域 VRAM 狀態或卸載所有模型。
- 依實際注意力實作與運算精度調整顯存需求估算。
- 將生成預覽更新移至採樣回呼，移除介面與 API 進度查詢中的重複更新。
- Refiner 卸載模型時加入找不到 UNet 的保護。
- 儲存 grouped INT8 量化權重時補齊 group_size 與 convrot_groupsize 參數。
- 減少 Lumina／SVDQ 不必要的張量複製操作；統一 Anima、Flux.2 與 Z-Image 的 Qwen 文字引擎屬性名稱。
- `comfy-kitchen` 更新至 0.2.37。
- 本整合版另外修正 Pillow、pillow-heif 與 protobuf 的依賴衝突，詳見「執行環境」。
- Intel XPU 的 Math SDPA 加入 query 分片，保留全長 K/V、遮罩與 GQA 行為，降低高解析度生成時完整注意力矩陣的記憶體尖峰；保留避免融合 SDPA 非有限輸出的相容性路徑。此為待 B580 實機驗證的候選修正。
- 驗證狀態：依賴政策測試 12 項通過；使用者回報 Windows／RTX 5060 Ti 啟動與生成正常，Intel Arc B580 啟動與基本生成正常，但高解析度修復第二輪採樣 OOM。XPU query 分片完成語法與 NumPy 模擬驗證，PyTorch 數值測試及 B580 高解析度修復重測仍待完成。本次僅更新測試分支，尚未發布為正式版。

### 新增

- 測試分支整合 Civitai Model Downloader v1.2.0，加入下載器、設定頁、狀態訊息與模型卡片操作的繁體中文翻譯。
- 補齊 LoRA 模型資訊視窗的訓練標籤、觸發詞、權重說明和卡片圖示提示文字。
- 首次安裝自動下載並驗證共用放大模型，提供下載進度條，且在內建放大、SD Upscale 與終極 SD 放大顯示繁中用途說明。
- 以 Forge Neo 最新 `neo` 分支為基礎建立 DianNaoTou 台灣繁體中文整合版。
- 加入完整 `zh_Hant` 語言檔，共 9,142 條介面翻譯。
- 全新安裝預設使用台灣繁體中文介面。
- 整合 ADetailer-Neo。
- 整合 forge-neo-image2prompt，包含本機相容性修正。
- 整合 sd-webui-prompt-all-in-one-neo，包含繁體中文翻譯。
- 整合 stable-diffusion-webui-tagger-fork，包含介面與相容性修正。
- 內建 Ultimate SD Upscale，保留原生 SD Upscale，並加入完整台灣繁體中文介面。
- 增加繁體中文 README，並保留上游英文原文。
- 新增 Intel Arc 通用啟動檔，沿用 Forge Neo 原始自動安裝流程並安裝 PyTorch XPU。
- NVIDIA 與 Intel Arc 使用獨立虛擬環境，避免 CUDA 與 XPU 套件互相覆蓋。
- Intel Arc 環境加入 XPU Math SDPA 相容性修正；Arc B580 為目前已驗證機型。
- 啟動器會以專案內的 `uv` 自動準備並鎖定 Python 3.13.12，不修改系統 Python、PATH 或 Registry。

### 上游同步

- 測試分支同步 Forge Neo 上游至 `e33f40e4`（2.29.1 開發分支）。
- LLLite ControlNet 配合 MultiDiffusion 分塊處理控制圖片；涵蓋 SDXL 與 Anima 路徑。
- 調整提示詞權重解析、LoRA Control、PNG Info 負面提示詞讀取及模型載入細節。
- 保留 `BREAK` 不算文字權重的判斷，避免 PNG Info 誤套用 Emphasis。
- MultiDiffusion 分塊設定加入固定介面識別碼；既有繁中翻譯可沿用。
- 補上 MultiDiffusion「從圖片自動偵測尺寸」按鈕提示的繁中翻譯。
- `README_EN.md` 更新至本次上游英文原文，繁中整合版說明維持於 `README.md`。
- 同步 Forge Neo 上游至 `41359cd4`。
- 新增提示詞輸入防抖與 img2img 保持長寬比模式。
- 整合模型載入、dtype、量化 metadata、RoPE、LoRA 與 ControlNet 修正。
- `comfy-kitchen` 更新至 0.2.35。

### 安全與容量

- 排除 Stable Diffusion、LoRA、VAE、Embedding、YOLO 等模型檔案。
- 排除使用者生成圖片、影片、`outputs`、`venv`、快取與備份檔。
- 插件所需模型仍依各插件原有機制在需要時下載。

### 執行環境

- 修正 2.29.2 上游合併後 requirements 與共用 constraints 衝突：沿用 Gradio 4.40 相容政策，統一 Pillow 10.4.0、pillow-heif 0.22.0 與 protobuf >=6.31.1,<7；新增跨檔鎖定一致性測試。

- 修正共用 `run_pip()` 將安裝專用參數附加到 `pip uninstall` 的問題，讓 CUDA／XPU 環境可正常移除衝突的 ONNX Runtime 套件。
- 跟隨 Forge Neo 最新上游需求。
- 目前上游測試環境為 Python 3.13.12。
- NVIDIA 使用 `venv-nvidia`，Intel Arc 使用 `venv-intel-arc`；兩者共用專案內的 Python 3.13.12 runtime。
- `uv` 下載支援斷點續傳、自動重試、Astral／GitHub 雙來源與 PowerShell 備援。
- NVIDIA 的 PyTorch／CUDA 套件由 Forge Neo 安裝器於首次啟動時自動安裝。
- Intel Arc 的 PyTorch XPU 套件由相同安裝流程於首次啟動時自動安裝。
