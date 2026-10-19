> **2026-10-19 維護決策：新的官方 XPU 為主線**
>
> `neo-test` 後續以 `webui-user-intel-xpu-official.bat`／`venv-xpu-official`（目前 Torch 2.14.1+xpu）為主要路徑。舊 Torch 2.14.0／`venv-intel-arc` 列為淘汰；Kitchen 社群方案已停止採用，專用啟動／建置入口已移除。
> **目前只留在測試版，等下一次大版本更新才納入正式 `neo`**。本次不刪除既有環境、不變更正式版，詳見 [決策紀錄](docs/decisions/2026-10-19-intel-official-baseline.md)。
>
> 測試啟動：`webui-user-intel-xpu-official.bat`。test.6 的採樣預設是原生 SDPA，三模式與回復方法見 [操作說明](docs/validation/xpu-sdpa-test6.md)。

## 舊路徑與 test.3 歷史說明（以下不再是測試主線安裝指引）

> test.3 的 Kitchen 比較環境已撤回；以下舊 Intel 2.14.0 內容僅供歷史參考，現在請使用上方官方啟動方式。

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
