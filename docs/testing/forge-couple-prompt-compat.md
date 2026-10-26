# Forge Couple 提示詞分區相容性測試（neo-test）

此項目僅在 `neo-test` 提供**選用、可還原**的本機修補工具；不修改正式分支，也不會自動覆寫第三方插件。

## 症狀

`[Forge Couple] ERROR - Not Enough Lines in Prompt... [1 / 3]`，接著可能於 `processing.py` 出現 `TypeError: 'NoneType' object is not iterable`。上游 v7.1.0 在 `after_extra_networks_activate` 使用 `kwargs["prompts"][0]` 取得提示詞，按使用者指定的 Separator 分割。若另一個流程已將它改寫為單段文字，即使原本輸入有區域分隔也會報錯。

## 使用（必須先關閉 Forge Neo）

在測試版根目錄使用 *NVIDIA 的 venv* 或電腦上的 Python 執行：

```bat
venv-nvidia\Scripts\python.exe tools\forge_couple_prompt_compat.py --check
venv-nvidia\Scripts\python.exe tools\forge_couple_prompt_compat.py --apply
```

擴充必須已安裝在 `extensions/sd-forge-couple`。工具只適用與已知 v7.1 原始碼相符的片段；版本不同會停止，不會猜測修改。套用前備份成 `forge_couple.py.before-dnt-fc-patch`，並先編譯檢查替換後的 Python 語法。

**此補丁是診斷性相容修正，非已完成實機驗證的問題根治。** 修補後先以 2 區、再以 3 區測試。若仍顯示 `[1 / 3]`，表示可取得的 `kwargs`、`p.prompt`、`p.all_prompts[0]` 都沒有正確分區，須追查其他提示詞插件的前處理順序；本修補不會猜測、憑空生出角色分區。Console 若出現 `Prompt compatibility fallback`，代表採用了較完整的候選提示詞來源。

**重要：** 要填寫與正面提示詞完全一致的 Separator。建議先用 `FC_REGION_2026`，共通首行＋三個角色間需要三次完整且相同的分隔字；這是診斷用文字，不是 Forge Couple 保留關鍵字。

回復：

```bat
venv-nvidia\Scripts\python.exe tools\forge_couple_prompt_compat.py --restore
```

如未安裝 NVIDIA 虛擬環境，可用相同版本的系統 `python` 取代；本工具只使用標準函式庫。若插件更新，先還原，更新後重新 `--check`。

## 測試矩陣

| 測試 | 預期 |
|---|---|
| Forge Couple 關閉 | 原生生圖正常 |
| Basic／2 區，無全域 | 不應再誤判成 1/2 |
| Basic／3 區，First Line 全域 | 4 段提示詞可辨識 |
| Hires.fix 啟閉 | 不應引入新例外 |
| 分隔字完全不存在 | 明確報數量不足，而不是偽造分區 |
| `--restore` | 可恢復原檔 |

LayerDiffuse 的 3/4 通道錯誤是**獨立問題**，本補丁不處理。測試時請先關閉 LayerDiffuse，以便確認 Forge Couple 自身結果；待兩者各自驗證後，再做共同啟用測試。
