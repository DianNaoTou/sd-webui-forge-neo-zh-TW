# Intel Arc B580 手動部署

此設定不會修改系統已安裝的 Python。所有套件均安裝在專案內的 `venv313`。

## 前置條件

- Windows 10/11 64-bit
- Intel Arc B580 與近期顯示驅動
- Git
- 獨立的 Python 3.13 runtime，放在專案上一層的 `Python313-runtime` 資料夾
- 至少一個 Stable Diffusion checkpoint，放入 `models/Stable-diffusion`

建議目錄：

```text
stable-diffusion/
├─ Python313-runtime/
└─ sd-webui-forge-neo-zh-TW-release/
```

## 安裝

在專案根目錄開啟 PowerShell：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-b580.ps1
```

腳本會：

1. 從旁邊的 `Python313-runtime` 建立隔離的 `venv313`。
2. 從 PyTorch XPU wheel index 安裝 `torch` 與 `torchvision`。
3. 安裝 WebUI 核心依賴。
4. 預先安裝 Prompt All-in-One 與 WD14 Tagger 所需依賴。
5. 驗證 `torch.xpu` 能辨識 Intel Arc。

## 啟動

```bat
webui-user-b580.bat
```

已驗證的 B580 參數：

```text
--skip-install --disable-sage --disable-flash --bf16-unet --fp32-vae --autolaunch
```

`--skip-install` 用來避免啟動器以 CUDA 套件覆蓋已安裝的 XPU 版 PyTorch。

## B580 NaN 修正

PyTorch XPU 的融合式 scaled dot-product attention 在 Arc B580 上可能間歇產生非有限輸出，使 WebUI 顯示：

```text
Encountered NaN in Latent
```

本分支在 `backend/operations.py` 中，僅針對 `xpu` 強制使用 `SDPBackend.MATH`。這會比融合式注意力稍慢，但能避開該正確性問題。NVIDIA 與其他裝置不受此分支影響。

參考：

- https://github.com/pytorch/pytorch/issues/192158
- https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html

## 已驗證環境

- Python 3.13.12
- PyTorch 2.14.0+xpu
- Intel Arc B580 12 GB
- BF16 UNet
- FP32 VAE
- SDXL / Illustrious checkpoint
- ADetailer-Neo
- WD14 Tagger（CPU）
- sd-webui-prompt-all-in-one-neo
