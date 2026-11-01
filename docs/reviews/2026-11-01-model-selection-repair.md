# SDXL / ANIMA 選模狀態修復與可回滾更新

基準：`neo-test`，`bd3604b5a3e900436c7f03b9e8bb0d66913dee17`，`neo 2.31-tw.1-test.5`。
修復分支：`fix/neo-test-model-state-rollback`。本 PR 不合併正式 `neo`，不變更發布版本或建立 Release。

## 已證實的原因

`_load_presets()` 先以 `refresh=False` 設定 dtype 與模組，再依賴 `checkpoint_change()` 更新載入參數。
checkpoint 相同時提前返回，導致 `shared.opts`、畫面和 `model_data.forge_loading_parameters` 不一致。
`forge_model_reload()` 的快取判斷與實際 loader 都讀取後者，所以空白欄位不能證明 Qwen 路徑已消失。

啟動時原本有兩個互不相依的 Gradio load event（建立選單、套用 Preset），且程式更新 Dropdown 不觸發 `.input()`；
因此必須先建立 choices，再套用和提交 Preset。啟動前的初始參數也必須使用目前 Preset，不能先載入殘留的全域模組。

## 修改清單

| 檔案 | 原因及行為 |
| --- | --- |
| `modules_forge/main_entry.py` | Preset 完整 tuple 一次發布、一次保存；相同 checkpoint/空白模組也更新 Preset 記憶與載入參數；`None` 正規化為空清單；依序初始化選單及 Preset；序列化選模/清單 callback；缺失選項維持可見並明確報錯，失敗不沿用舊參數。 |
| `modules/sd_models.py` | 單次載入使用同一份參數及模組清單快照，快取 hash 只描述實際交給 loader 的內容。 |
| `modules/sysinfo.py` | API/override dtype 改變也更新載入參數；修正 checkpoint 分支跌入一般設定 setter 的 `if/else` 結構。 |
| `modules_forge/patch_basic.py` | 任何載入例外均保留原始檔與既有 `.corrupted` 備份；保留真正例外鏈，取消自動改名/刪除。 |
| `tests/test_model_selection_sync.py` | 實際函式的 Preset、空值、殘留路徑、dtype/API、loader/cache 邊界與模型檔保護回歸。重量模型/GPU/UI 依賴以替身隔離。 |
| `tests/test_additional_module_paths.py` | 真正缺檔/歧義必須報錯，不再把「略過檔案」當成功。 |
| `tools/repair-model-selection.ps1` | Windows 入口，固定完整修復 SHA，檢查仍在執行的專案 Python 程序，不會自動終止程序。 |
| `tools/repair_model_selection.py` | 四個程式檔的白名單更新、原始位元組及 SHA-256 備份、Junction 清單、套用後自動回歸、失敗還原、受保護首次啟動。 |
| `tests/test_model_selection_repair.py` | 真實檔案與 Git fixture 驗證套用/回滾、部分失敗、個人修改、備份毀損、模型隔離；Windows 原生 Junction 測試；環境路徑測試不實際安裝套件。 |
| `.github/workflows/cpu-regressions.yml` | Windows 加入同步/回滾測試與 PowerShell 語法解析；Linux 一般/最佳化完整套件仍執行新測試。 |

## Windows 使用

先關閉 Forge。取得 PR 最終已審查的完整 commit SHA，下載該 SHA 下的
`tools/repair-model-selection.ps1`，然後在 PowerShell 執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\forge-model-repair.ps1" -Root 'D:\sd-webui-forge-neo-zh-TW-test' -Revision '<完整修復 commit SHA>'
```

這不是 `git pull` 或分支切換：只將四個修補檔套用到既有 `neo-test`，本機 HEAD 保持原值。
為避免未審查的版本混用，目前只接受上述基準 commit；目標程式檔有個人修改、分支不同、路徑不明或版本不同時，
在寫入前停止。其他檔案與未追蹤檔案不會清理。未解決的基準差異需要重新審查，不能以強制 reset 繞過。
程式檔備份會保留 Windows 原始換行和實際位元組。

可用 `-NoLaunch` 只套用並執行回歸，不啟動。可用 `-Python` 指定既有 Python；
預設先尋找專案 managed Python，再嘗試官方 XPU venv、一般 venv/PATH；只使用能執行的 Python 3.13，損壞的 venv 不會被當作修復工具的唯一入口。
`-ModelRoot` 可明確指定既有模型根目錄；未指定時檢查專案 `models`，並記錄實際解析位置及所有下層 Junction。
更新器本身不會開啟權重雜湊、搬移、刪除、覆蓋或下載模型；啟動 Forge 後正常選模仍會讀取權重及模型 metadata。

### 環境及首次啟動

既有官方 XPU 環境能 import 核心依賴且 Torch 為 `2.14.1+xpu`、XPU 可用時直接沿用；
否則在外部備份目錄建立另一個 venv，安裝專案 requirements 與固定 XPU Torch/torchvision。
`-RebuildEnvironment` 可明確要求建立替代環境。舊 venv 不會刪除或搬移。
替代環境的完整 Windows 安裝仍待實機驗證；失敗自動還原程式檔並保留安裝紀錄。

首次啟動使用設定副本，原始設定、styles、venv 不會被更新器覆寫；
使用 `--skip-prepare-environment --skip-install --disable-all-extensions`，HF/Transformers offline，避免首次驗證觸發安裝或外掛模型下載。
因此首次保護啟動不含 WD14、ADetailer、ControlNet、LayerDiffuse 等外掛功能，這不是永久修改外掛設定。
套用成功後，既有 Intel Arc 啟動入口也會使用修補後的程式，但其原有套件安裝/外掛行為不屬於保護啟動。
替代環境只有保護啟動會使用，原有 launcher 不會自動改用它。

### 回滾

開始套用前即顯示 `%LOCALAPPDATA%\ForgeNeoRepair\<交易目錄>` 與完整回滾指令。
備份內含原始程式、manifest（來源 HEAD/分支、Git 狀態、檔案前後 SHA-256、模型/Junction 位置）、
獨立 `rollback.ps1` 及離線還原 helper；啟動/安裝輸出記在 `launch.log`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File '<交易目錄>\rollback.ps1'
```

先關閉 Forge。還原前先檢查所有原檔備份及目前程式雜湊；若修補後又有個人修改、備份毀損或 Junction 位置改變，
停止而不覆蓋。部分套用失敗或啟動程序以錯誤碼結束時自動還原原始程式；斷電/強制關閉後可使用同一還原入口重試。
還原不需要網路或切分支，不刪除任何替代環境、設定副本、輸出或模型。
回滾恢復的是執行前版本，原版本的 bug 也會恢復。日後一般 Git 更新前應先回滾這份 overlay，避免衝突。

## 驗證與限制

- 基準原始碼跑新增同步測試：14 項中 8 failure、5 error；關鍵空白模組/相同 checkpoint 殘留測試確實失敗。
  後續另增加 dtype-only API 測試；上述基準數字是增加該項之前的紀錄。
- 本地同步/回滾及既有路徑、XPU profile、settings-version 合計 **50 項，47 通過、3 項 Windows 專用跳過**。
  新增同步/回滾另以 `python -O` 執行 **28 項，27 通過、1 項 Windows Junction 跳過**。修改的核心 Python 檔案編譯通過。
- Windows 原生 Junction、PowerShell 解析和既有完整測試交由 PR CI 驗證；本地沒有 PowerShell/CPU Torch，
  `test_forge_231.py` 在本地因缺少 Torch 無法匯入，不能宣稱完整既有套件已通過。
- 尚未讀取使用者實機的完整 traceback/config；已證明的缺陷不代表已排除所有外掛/API/瀏覽器事件情境。
- **未存取使用者 A380、未讀取 100GB 模型、未完成 Windows/XPU 安裝或正常出圖實機驗證。**
  ANIMA 所需的模組若確實不存在，仍應顯示缺檔；修補不會下載或隱藏它。
