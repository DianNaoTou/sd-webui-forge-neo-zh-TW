# 2.30-tw.2-test.2：ControlNet 錯誤中止與生命週期修補

2026-10-18（Asia/Taipei）｜ChatGPT

使用者試跑 test.1 後授權修補並更新測試版。本輪以 neo-test `974c8d4f4182accc88b0f6c41cd224751331a90a` 為基礎，只更新 neo-test，不建立正式版 Tag／Release、不修改正式 neo。

## 試跑證據與可確認的原因

- RTX 5060 Ti／torch 2.13.0+cu130 正常啟動，多次基本生成及兩段採樣完成。
- CLIP-ViT-H (IPAdapter) 輸出 dict，卻被一般 ControlNetPatcher 接收，進入 image_resize 時發生 `'dict' object has no attribute 'shape'`。搭配錯誤可確認，僅依一般檔名不能判定權重原始來源。
- animal_openpose／InstantID 有未選控制模型的錯誤。原先初始化失敗被 ScriptRunner 一般例外處理攔下後繼續，後續回呼直接索引 current_params[0]，產生連鎖 KeyError。
- ControlNet 警告反覆出現。原碼快取只保留 nn.Module，而每次 ControlNet.copy 都建立獨立 ModelPatcher；LoadedModel 對 ModelPatcher 為 weakref。包裝器死亡但底層模型仍被快取保留時，正好符合 is_dead 的告警條件。
- 原先採樣成功才會走 sampling_cleanup，任意採樣失敗沒有覆蓋整個 process_images 的擴充清理。本次增加 finally 清理入口，避免只依賴成功的 postprocess。

## 變更

1. modules/scripts.py 新增 ScriptAbortError；process／sampling 回呼明確收到此錯誤時向上傳遞，停止該次生成。其他擴充的一般例外處理保持原狀。加入可選 cleanup_processing(p) 方法，逐一清理且個別失敗不擋住其他清理、不蓋掉原錯誤。
2. modules/processing.py 在 process_images 的 finally 呼叫清理，涵蓋初始化、生成／採樣、後處理失敗及成功；不依賴 GUI Stop，不設定跨請求的全域 interrupted 旗標。
3. ControlNet 在預處理／下載前檢查模型是否已選、存在且可辨識，並對已知 IPAdapter／一般 ControlNet 型別提前檢查；對一般 ControlNet 要求 tensor、IPAdapter 要求含 image 的特徵 dict。不依賴檔名猜測模型型別；未宣告資料契約的第三方 patcher 保留既有行為。
4. 失敗轉成帶單元編號與原因的錯誤，清除所有已建部分狀態。採樣缺少初始化不再直接索引 KeyError；後處理不碰缺失的參數。成功及失敗最後都清除模型／預處理輸出／遮罩引用，清理目前 ControlNet 並恢復掛載前 UNet；清理可重複呼叫。
5. 清理以請求 ID 區分；ADetailer 的 control_net_disabled 巢狀請求不清除外層狀態。修補保留多單元與兩段 HR 回呼流程。任意第三方「啟用 ControlNet 的巢狀請求」並非本輪宣稱已支援的範圍。
6. 一般 ControlNet 快取保存模型及其顯存包裝器；同權重複本共用包裝器、條件狀態仍獨立。快取鍵加入裝置、儲存與計算 dtype，避免跨環境誤用。清理不再把共用包裝器從記憶體管理器移除，管理器仍能追蹤及按需卸載快取中的權重；沒有關閉洩漏警告，也沒有修改全域 GC／顯存政策。ControlLoRA 的 cleanup 改成 prepare 前與重複呼叫皆安全。
7. 版本檔、README、RELEASES、CHANGELOG 標為 2.30-tw.2-test.2。保留所有 test.1 上游整合、繁中、NVIDIA／XPU 與擴充功能。

## 測試

Linux／Python 3.12／torch 2.14.1+cpu；測試工作機依賴，不改專案 CUDA／XPU PyTorch 要求。

- 完整 unittest：79 tests 通過、0 skip。
- 新增 test_controlnet_lifecycle.py：13 tests 通過；Python -O 同樣 13/13。
- 既有 VAE／Flash 的 Python -O 測試：5/5 通過。
- compileall：backend、modules、modules_forge、scripts、tests、ControlNet／IPAdapter 擴充成功。
- git diff --check（保留 CRLF 的檔案使用 core.whitespace=cr-at-eol）通過。
- CI 增加兩個擴充編譯及 ControlNet -O 測試；遠端結果記於 ai-work-log Issue #5。

新增測試涵蓋：初始化第二單元失敗清空所有部分狀態；多單元／HR 回呼；採樣掛載錯誤後可重試；ADetailer 停用的巢狀請求不清父請求；缺參數明確停止；清理自身失敗仍恢復 UNet；runner 只傳遞明確 abort、保留其他例外策略；真實 finally 在成功／採樣失敗皆執行；型別契約；未選與錯配在呼叫預處理器前攔截；真實 copy／cache loader 的包裝器共享、dtype 隔離、weakref 保活與淘汰後釋放；ControlLoRA 在 prepare 前及重複清理的安全性。

測試使用 AST 隔離實際 production 方法，將 GPU／UI 服務替換成 CPU 或 mock。涵蓋生產函式邏輯，但不等於已完成全套 WebUI 匯入、模型 GPU 權重或 Windows／Gradio 測試。初次測試因 stub 未接收原始控制單元的額外參數而失敗，修正測試 stub 後重跑成功；沒有因此更改產品介面。

## 必須由使用者重新驗證

關閉 WebUI／Python 程序，再在 neo-test 執行 git pull --ff-only，完整重啟；確認顯示 neo 2.30-tw.2-test.2，不能只按 Reload UI。

1. ControlNet 全關：一般生成與 HR，確認既有結果／速度沒有明顯回歸。
2. 故意啟用單元但模型選 None：應顯示該單元錯誤並中止，不應產生新的無控制成圖、KeyError 或繼續採樣。
3. 故意將 CLIP-ViT-H (IPAdapter) 搭一般 ControlNet 權重：應在第 0 步前清楚拒絕，不再進入 dict.shape traceback。
4. 改成配套的 SDXL ControlNet／預處理器，及配套 IPAdapter 權重／預處理器：確認實際控制效果，不只看進度 100%。
5. 先故意失敗，再修正設定直接生成；連續至少 3 次，觀察警告及工作管理員顯存，不須每次重啟。
6. 批次 1／2、兩個控制單元、HR 開關、ADetailer 開啟的常用流程。
7. Arc B580 先用小尺寸測相同流程；CUDA／XPU 本輪都未實測。

GPU 洩漏警告是否完全消失、模型效果與速度，需上述實測確認。本輪修的是已辨認的包裝器生命週期及失敗清理；不是保證所有 ControlNet／IPAdapter／InstantID 版本、CLIP 架構及 SD1/SDXL 模型皆相容。模型配錯會被拒絕，不會自動轉換成相容權重；前一版已殘留的狀態需完整重啟後再測。
