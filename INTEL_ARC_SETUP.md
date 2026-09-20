# Intel Arc 安裝與啟動

Intel Arc 版本沿用 Forge Neo 原本的自動安裝流程。第一次啟動時會建立獨立的
`venv-intel-arc`，自動安裝 PyTorch XPU、Forge Neo 核心相依套件及內建插件的相依套件。
啟動器也會在專案內自動準備固定版本的 Python 3.13.12，不使用或修改系統 Python。

## 前置條件

- Windows 10／11 64 位元
- Intel Arc 顯示卡與近期驅動程式
- Git
- 至少一個 Stable Diffusion checkpoint，放入 `models\Stable-diffusion`

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

## 已驗證環境

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
