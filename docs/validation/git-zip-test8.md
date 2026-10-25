# Test 8：ZIP 直接使用 Git 更新

版本：neo 2.30-tw.2-test.8，2026-10-23。只更新 neo-test，正式 neo／正式 Release 不變。

## 設計與範圍

Release 附件 neo-test.zip 包含程式與完整可攜的 depth 1 單分支 Git 儲存庫，追蹤 origin/neo-test。
不附主模型、Python、venv、使用者設定或輸出。使用者需已安裝 Git；不需要初始化腳本。
同一包裝工具支援 neo，但本輪不發布正式包，也不將測試包偽裝成追蹤正式版。
GitHub archive/refs/heads/neo-test.zip 與 Source code (zip) 維持原始碼封存，沒有 .git。

發布沿用現有 CPU regressions 工作流，push neo-test 時等待 Linux 與 Windows 兩個工作成功，
再以被驗證的精確 SHA 打包。建立固定版本 Prerelease、ZIP 與 SHA-256，不更新正式 Latest。
已發布的相同版本保留；失敗的草稿僅在 Tag 與 SHA 相符時續作，不覆寫既有附件。

## 使用與保護

解壓到新的資料夾，保留隱藏的 .git。先關閉 WebUI，再於解壓後專案根目錄執行：

```bat
git branch --show-current
git pull
```

分支應為 neo-test。同一資料夾可沿用，下載新 ZIP 並不代表必須每次另裝環境。
請勿用 ZIP 蓋到現有安裝或舊 .git。舊 archive ZIP 不會自行變成 Git 儲存庫；本輪沒有原地轉換舊 ZIP。

預設 pull.ff=only、pull.rebase=false、merge/rebase.autoStash=false，
branch.neo-test.mergeOptions=--no-overwrite-ignore。普通 git pull 在歷史分歧、受追蹤檔案修改衝突、
或 ignored 同名檔案將被覆寫時停止，保留原檔；不使用 reset --hard、clean、skip-worktree 或 assume-unchanged。
這些是封裝內的本地設定，使用者自行覆寫設定或用強制指令不在此保護範圍。

模型／輸出／根設定／runtime／venv 不受追蹤；包裝檢查禁止提交使用者資料，僅允許既有模型說明佔位檔。
隨附插件程式仍會更新，自裝插件保留。webui-user.bat／webui-user.sh 雖列在 ignore 仍是受追蹤檔案，
自行修改後可能阻擋更新，不能宣稱 ignore 能保護它們。NVIDIA 個人覆寫可沿用 webui.bat 已支援的
webui.settings.bat，本輪將它明確忽略；不新增 Intel 個人覆寫機制。

單純 git pull 不執行 pip、不重抓模型、不建立 venv。本輪未改啟動或下載邏輯；更新後啟動仍依原機制
處理缺少／不符的依賴。Test 7 的 Intel 官方環境使用 venv-xpu-official；舊正式版只有 venv-intel-arc
時首次使用測試版仍需建立官方環境，且獨立 HF 快取可能需要下載缺少的檔案，不能當作本輪修復。
Test 7 原有舊設定確認／備份流程維持，ZIP 更新本身不備份資料。

## 測試

自動測試使用臨時本地 Git 來源、實際 ZIP 解壓與普通 git pull：

- 固定舊 SHA，即使遠端 tip 已前進仍包裝正確版本；搬到不同目錄後 Git 狀態乾淨、fsck 成功。
- neo／neo-test upstream 正確，包含 shallow 資訊，沒有外部 objects、憑證或 hooks。
- 跨兩次提交更新，模型、設定、輸出、插件、Python runtime 與 venv 原 bytes 保留。
- 修改受追蹤啟動檔、提交本地分歧、遠端新增 ignored 同名檔案時停止；不自動 stash。
- 拒絕打包追蹤的個人設定、不覆寫先前封裝、校驗 SHA-256。

Linux 執行完整既有回歸；Windows 執行原有檔案 I/O／設定／XPU 隔離與新增 ZIP Git 測試。
CI 包裝實際專案後再次解壓，核對 HEAD、upstream、乾淨狀態與 fsck。
容量由包裝工具報告 zip_bytes、source_compressed_bytes、git_compressed_bytes，為實測壓縮資料量；
後兩者不含 ZIP 目錄／檔頭開銷，不與 GitHub archive 大小混為一談。

仍待人工驗證：Windows 內建解壓與 7-Zip、含中文／空白的路徑、實際上一測試 ZIP 到未來提交的 GitHub 更新、
RTX 5060 Ti／B580 的啟動與生成／HR、Test 7 重複上傳與圖像資訊傳送、主要插件。
CI 與本地 Git 測試不是 GPU 推論驗收，不宣稱修復所有 OOM／DEVICE_LOST／權限問題。

## 本輪本機結果

隔離 Linux／Python 3.12／真實 CPU PyTorch 2.14.1+cpu、固定 Gradio 4.40／FastAPI 0.127.1：
147 項 Python 回歸全部通過、零跳過；新增 8 項真實 Git／ZIP 測試已包含其中。
59 項 Python -O 測試、4 項 Node 前端測試與核心／插件編譯通過。Windows CI 與發布附件另以遠端執行結果為準；未做 GPU 生圖驗收。
