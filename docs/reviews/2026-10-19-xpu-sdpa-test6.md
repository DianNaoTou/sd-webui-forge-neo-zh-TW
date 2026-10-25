# test.6：可回復的 XPU 原生 SDPA 實驗

## 目的與歷史證據

使用者希望測試新 PyTorch 下，以前為避免記憶體爆掉而加入的慢速處理是否
仍然需要。過往對話搜尋這次未能取得完整對話；以下依 Git 原始修改與當時
CHANGELOG 定位，不冒稱已重讀全部歷史對話。

- `63bad99ea93e5cfb9bce422c3f6e1f68152106b3`（2026-09-13）：
  為 Intel Arc 融合 SDPA 間歇性非有限輸出，強制 `SDPBackend.MATH`。
- `20a10e3d3846bec7d4f13eed2b65c026ee98bded`（2026-10-04）：
  當時紀錄為 B580 基本生成正常、HR 第二輪採樣 OOM；加上 query slicing，
  每片最多 256 列、估算 128 MiB 注意力暫存預算，仍強制 Math。
- 上述是注意力運算 workaround，不是模型按需載入／卸載政策。
  目前沒有證據支持回退整套 `memory_management.py` 或取消所有顯存保護。

## 最小修改

test.6 新預設 `--xpu-sdpa native`，XPU 路徑直接交給目前安裝的 PyTorch
`scaled_dot_product_attention` 自行選擇後端；不套本地 Math context、不分塊。
保留 `math`（舊的不分塊 Math）與 `sliced`（test.5 原有 Math 分塊）供對照。
沒有自動 fallback／重試，實驗中的 OOM／kernel error／NaN 不會被偷偷換模式
掩蓋。未知模式由 CLI／dispatcher 拒絕。

既有 masked GQA 的 K/V 展開邏輯不變。修改只在張量裝置為 `xpu` 時選擇路徑；
原本 NVIDIA CUDA 優先序實作原封不動。`--xpu-sdpa` 不會改變 CUDA 行為。
首次實際呼叫記錄 mode、Torch 完整版本與裝置；XPU profile 的啟動報告與
`runtime-report.json` 也記錄 mode，不需等生圖才能確認啟動選項。

沒有修改 UI、CSS、翻譯、插件、PyTorch／TorchVision／Kitchen 依賴、
模型卸載管理、Never OOM、VAE tiling、顯存預留或各 profile 的 venv／設定位置。
保留 test.5 UI 修復、ControlNet 生命週期修補及 Kitchen 版本資訊修補。
版本識別更新為 `neo 2.30-tw.2-test.6`；僅提交 `neo-test`，不發布 Release。

## 修改檔案

| 檔案 | 內容 |
|---|---|
| `backend/args.py` | XPU 專用三模式 CLI，native 預設 |
| `backend/operations.py` | 僅 XPU 分派到選定模式；CUDA 路徑不變 |
| `backend/xpu_attention.py` | 原生／舊 Math／舊分塊分派與一次性日誌 |
| `tools/launch_xpu_profile.py` | 啟動報告記錄模式 |
| `modules_forge/forge_version.py` | test.6 版本識別 |
| `tests/test_xpu_attention.py` | 新增 8 項模式、CLI、數值與錯誤傳播回歸 |
| `tests/test_xpu_profiles.py` | 新增 1 項對照參數不改 profile／環境的回歸 |
| `.github/workflows/cpu-regressions.yml` | 新增最佳化模式注意力回歸 |
| `docs/validation/xpu-sdpa-test6.md` | 三模式操作、回復與 B580 對照表 |
| `docs/reviews/2026-10-19-xpu-sdpa-test6.md` | 本次工程紀錄 |
| `CHANGELOG.md`、`README.md` | 測試版紀錄與操作入口 |

## 驗證

- Python 3.12、Torch 2.14.1+cpu：完整 **125 項 unittest 通過，0 skipped**。
- Gradio 4.40.0／FastAPI 0.127.1／Pydantic 2.10.6 的 UI 資源 HTTP 回歸，
  UI Freeze、翻譯、插件原有回歸均通過。
- Node：既有 **4 項 WD14 前端事件測試通過**。
- `python -O`：注意力與 XPU profile **26 項通過**。
- compileall、保留 CRLF 的 Git 差異檢查通過。
- 模式測試包含：native 不建立 Math context／不呼叫 slicing、只呼叫一次；
  Math 不分塊；sliced 委派既有函數；遮罩與 scale 參數保留；OOM／kernel
  error 原樣傳播，NaN 不自動重試；CLI 預設／明確選項／非法模式；
  XPU 分派與 masked GQA、CPU／CUDA 一般分支不變；profile 參數轉送。

數值驗證使用真實 CPU Torch；XPU／CUDA 分派契約使用裝置與服務替身。
尚無 RTX 5060 Ti／B580 實機測試，不宣稱新核心已改善顯存、效能或穩定性。

## 風險與回復

native 可能重新出現 NaN、黑圖或 OOM；Math 取消分塊亦可能重現 HR OOM。
這是刻意暴露原路徑的測試版本。`native` 只代表交給 PyTorch 選擇，不能從
模式名稱推斷實際用了融合核心。若模型／設定走 CK 等不同注意力入口，這項
測試可能沒有被觸發，應以首次呼叫日誌為依據。

直接加 `--xpu-sdpa sliced` 重啟便可回到 test.5 注意力保護，模型、UI 與 venv
不需要還原。三模式的完整比較與命令見
[實機清單](../validation/xpu-sdpa-test6.md)。整個提交亦可一般 `git revert` 還原。
