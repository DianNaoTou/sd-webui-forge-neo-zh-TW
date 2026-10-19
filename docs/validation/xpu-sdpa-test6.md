# test.6：新版 PyTorch 的 XPU 原生注意力對照測試

## 這次恢復的是什麼

Git 紀錄顯示兩層本地 workaround：

| 修改 | 原因 | test.6 對照模式 |
|---|---|---|
| `63bad99e`（2026-09-13）強制 Math SDPA | Intel 融合注意力曾產生非有限輸出／NaN | `math`：保留此限制，但不分塊 |
| `20a10e3d`（2026-10-04）Math query 分塊 | B580 的 HR 第二輪採樣 OOM；將注意力矩陣分批計算 | `sliced`：保留 test.5 的完整保護路徑 |
| 取消上述兩項注意力限制 | 讓目前安裝的 PyTorch 自行選擇 SDPA 後端 | `native`：test.6 的實驗預設 |

此處的「分塊」是注意力 query 的分批運算，不是模型依需求從 RAM 載入／卸載。
本次不修改 `backend/memory_management.py`、Never OOM、VAE tiling 或使用者
已有的顯存設定；不將兩種不同問題混在同一個測試變因中。

`native` 並不保證使用 Flash／融合核心、提升速度或改善顯存。
PyTorch 仍可根據裝置、dtype、形狀、遮罩選擇自己的 Math 實作。
這版用來確認原本的 NaN／OOM 是否仍可重現，不能先宣稱已修好。
原生路徑失敗時不自動重試 Math／分塊，避免隱藏實際測試結果。

## 更新與執行

在目前使用的測試目錄確認分支是 `neo-test`，保留本機修改，然後：

```powershell
git pull --ff-only origin neo-test
```

使用相同 official XPU 環境依序測三次；每次關閉前一次 WebUI，再重啟。
以下均沿用相同 venv、設定、模型、連接埠與 PyTorch，不需要刪除環境。

```powershell
# test.6 預設：原生 PyTorch SDPA
.\webui-user-intel-xpu-official.bat

# 舊狀態一：強制 Math，但取消 query 分塊
.\webui-user-intel-xpu-official.bat --xpu-sdpa math

# 舊狀態二：回到 test.5 的 Math + query 分塊保護
.\webui-user-intel-xpu-official.bat --xpu-sdpa sliced
```

也可明確傳 `--xpu-sdpa native`。已建置好 kitchen profile 者，可把檔名換成
`webui-user-intel-xpu-kitchen.bat`，其餘相同；不要為此測試另外安裝 kitchen。
舊 `webui-user-intel-arc.bat` 使用者，預設同樣是 native；要對照時將
`--xpu-sdpa math` 或 `--xpu-sdpa sliced` 加到該檔原有 `COMMANDLINE_ARGS`。

profile 的啟動報告與 `runtime-report.json` 會列出 `xpu_sdpa`。
第一次實際執行該注意力路徑時，日誌顯示：

```text
XPU SDPA mode=native | PyTorch=... | device=xpu:0 | no automatic mode fallback
```

如果從未看到此訊息，可能模型／設定使用了不同注意力實作，不可認定這次
生成已測到該模式。保留原本 PyTorch cross attention 設定，不開 CK／其他
注意力後端當成同一次對照；遇到此情況記錄啟動參數與模型後再判讀。

NVIDIA RTX 5060 Ti 的 CUDA 路徑不受 `--xpu-sdpa` 影響；
它可以驗證 test.6 的既有 UI／生成是否正常，不能用來證明 XPU workaround
已可移除。UI、繁中、插件、ControlNet 修補及所有套件版本均保留。

## B580 每個模式使用完全相同參數

優先使用之前確實會 OOM 的模型、解析度、放大器、HR 倍率與 steps。
若記不得原參數，先選固定的基準組合，再逐步提高到原常用 HR 條件。

1. 記錄 commit／test.6、Torch 完整版本、驅動、SDPA mode、模型與 VAE。
2. 固定種子、batch=1、相同提示詞、採樣器、步數與所有顯存設定。
3. 基本生圖先測一次，再用相同圖設定開 HR；連續生成至少三次。
4. 記錄第一輪、HR 第二輪及 VAE 階段耗時；觀察 GPU 專用／共用記憶體、
   系統 RAM、輸出黑圖／NaN、OOM、第二三次生成是否比第一次更容易失敗。
5. 控制插件狀態一致；先完成純生圖對照，再另測 ADetailer／ControlNet。
6. 測完 native 重啟切到 math，再重啟切到 sliced；保留相同參數。
   同模式內記錄首次與後續生成，避免把首次編譯／載入成本算成穩態速度。

| 模式 | 基本生圖 | HR／連續三次 | 第一輪／HR 耗時 | 專用／共用 VRAM、RAM 峰值 | NaN／黑圖／OOM |
|---|---|---|---|---|---|
| native（新預設） | 未測 | 未測 | 待填 | 待填 | 待填 |
| math（取消分塊） | 未測 | 未測 | 待填 | 待填 | 待填 |
| sliced（test.5 保護） | 未測 | 未測 | 待填 | 待填 | 待填 |

若只有 native 成功且更快，代表當前模型／參數下原 workaround 可能已不需要；
仍不能推論所有模型已修復。如果 math 仍 OOM、sliced 成功，則原先的記憶體
保護仍有作用。若三種都慢，再排查模型卸載、RAM／共用顯存、VAE 或插件，
不繼續盲目回退整個記憶體管理核心。

## 回復

若 native 出現 OOM／NaN，關閉 WebUI 後以 `--xpu-sdpa sliced` 重啟，
即可使用原保護路徑，不需要換分支、降級 Torch 或重建 venv。
此設定是啟動參數；普通雙擊啟動器仍會使用 test.6 預設 native。
