# Intel Arc 安裝與啟動

`neo-test` 的 Intel Arc 現在統一使用官方 PyTorch XPU：

```bat
webui-user-intel-arc.bat
```

僅保留 `webui-user-intel-arc.bat` 一個 Intel 啟動入口，直接執行官方 XPU 流程，支援額外參數。原 `webui-user-intel-xpu-official.bat` 已移除。

- Python：專案專用 3.13.12，由啟動器準備，不修改系統 Python。
- 依賴：Torch 2.14.1+xpu／TorchVision 0.29.1+xpu。
- 虛擬環境：`venv-xpu-official`，與 NVIDIA 獨立。
- 設定／快取／輸出：`runtime/xpu-ab/official`，沿用既有官方組設定，不覆寫。
- 模型：共用專案 `models`，無需重新下載。
- UI、繁中及插件操作方式共用原本整合版。
- 原 `venv-intel-arc` 不再建立或啟用；不自動刪除既有環境與資料。
- 舊根目錄設定僅在官方設定尚不存在時複製；已有官方設定不會被舊設定蓋回。

需要 Windows 10／11 64 位元、Git、Intel Arc 與相容驅動。首次啟動會建立官方環境、安裝依賴並驗證 XPU；已有官方環境可直接沿用。預設網址為 http://127.0.0.1:7862 。

## B580 建議操作與驗證

採樣預設原生 SDPA。使用者已回報開啟「Never OOM 記憶體最佳化 → 對VAE啟用（始終分塊）」後，原先 HR 結束階段的嚴重記憶體壓力明顯改善；「為 UNet 啟用（始終卸載）」保持不勾。
此選項由使用者在 UI 選擇，本次沒有強制改寫預設值。此次簡化仍沿用官方組設定；請確認勾選狀態。

更新後請驗證基本生圖、原本 HR 參數連續兩次、VAE 是否仍 OOM，以及生成後系統反應。A380 與其他 Arc 尚未在本輪驗證。
原生／Math／分塊三模式對照見 [test.6 操作說明](docs/validation/xpu-sdpa-test6.md)。

**僅修改 `neo-test`；仍等下一次大版本才納入正式 `neo`，不發布 Release。**
Kitchen 社群 fork 支援已移除；主程式必要的官方 comfy-kitchen 依賴保留。
詳見 [維護決策](docs/decisions/2026-10-19-intel-official-baseline.md)。

---

## 歷史安裝說明（已由上方官方路徑取代，請勿照此建立舊環境）

以下保留舊 2.14.0 的調整背景與紀錄，並非目前啟動器行為。

# Intel Arc 安裝與啟動

Intel Arc 版本沿用 Forge Neo 原本的自動安裝流程。第一次啟動時會建立獨立的
`venv-intel-arc`，自動安裝 PyTorch XPU、Forge Neo 核心相依套件及內建插件的相依套件。
啟動器也會在專案內自動準備固定版本的 Python 3.13.12，不使用或修改系統 Python。

## 前置條件

- Windows 10／11 64 位元
- Intel Arc 顯示卡與近期驅動程式
- Git
- 啟動 UI 不需要 checkpoint；需要主模型推論時才將 checkpoint 放入 `models\Stable-diffusion`

## 安裝與啟動

在專案根目錄執行：

```bat
webui-user-intel-arc.bat
```

第一次啟動會自動：

1. 將專案專用的 `uv` 與 Python 3.13.12 安裝到 `runtime`。
2. 使用該 Python 建立獨立的 `venv-intel-arc`。
3. 從 PyTorch XPU wheel index 安裝 `torch` 與 `torchvision`。
4. 安裝 Forge Neo 核心及內建插件所需的相依套件。
5. 確認 PyTorch 能辨識可用的 XPU 裝置。
6. 啟動繁體中文 Forge Neo。

Intel Arc 與 NVIDIA 使用不同的虛擬環境，因此不會互相覆蓋 PyTorch 套件。
專案內 Python 不會加入 PATH 或 Windows Registry；電腦原有的 Python 版本不受影響。

## 預設參數

```text
--disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch
```

- SageAttention 與 Flash Attention 的預編譯套件目前以 CUDA／NVIDIA 環境為主，因此在 Intel Arc 啟動檔中停用。
- BF16 UNet 可降低顯存用量。
- FP32 VAE 可降低部分模型在 XPU 上產生 NaN 的機率。

## XPU Math SDPA 相容性修正

部分 Intel Arc 與 PyTorch XPU 組合使用融合式 scaled dot-product attention 時，可能間歇產生非有限輸出，使 WebUI 顯示：

```text
Encountered NaN in Latent
```

本整合版在 `backend/operations.py` 中，僅針對 XPU 強制使用 `SDPBackend.MATH`。
這可能比融合式注意力稍慢，但能優先確保輸出正確；NVIDIA 與其他裝置不受影響。

參考：

- https://github.com/pytorch/pytorch/issues/192158
- https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html

## 先前使用者實測環境（不代表本次修正已通過實機驗收）

- Windows 10
- Python 3.13.12
- PyTorch 2.14.0+xpu
- Intel Arc B580 12 GB
- SDXL／Illustrious checkpoint
- ADetailer-Neo
- WD14 Tagger（CPU）
- sd-webui-prompt-all-in-one-neo

Arc B580 是目前實際驗證的機型，不代表此啟動方式僅支援 B580。其他 Intel Arc
型號仍取決於顯示驅動程式與 PyTorch XPU 對該硬體的支援狀態。

## 本次全新安裝驗收

請依 [INSTALLATION_TESTS.md](INSTALLATION_TESTS.md) 建立全新測試副本，確認依賴、空 checkpoint 啟動、首次下载和第二次啟動。
