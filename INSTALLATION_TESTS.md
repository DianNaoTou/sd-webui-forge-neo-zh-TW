# 全新安裝修正與驗收（dianaotou-zh-tw）

本次修改只供測試分支。未合併或推送至 `neo`。

## 修正原因與檔案

| 檔案 | 修改與原因 |
| --- | --- |
| `requirements.txt` | protobuf 改為 `>=6.31.1,<7`；Hub 改為 `>=0.34.0,<1.0`；Pillow 改為 10.4.0，HEIF 改為 0.22.0；明列 Gradio 4.40.0／rangeslider 0.0.8。 |
| `constraints-common.txt`（新增） | 限制擴充套件安裝時的共通相容範圍，包括 ONNX／TensorFlow，避免後續安裝再次拉高 Pillow／Hub 或降回舊 protobuf。 |
| `modules/dependency_utils.py`（新增） | 使用標準 Requirement parser 檢查精確版本、上下限、marker、未固定版本及 extras；依 PyTorch build 選擇 ONNX Runtime，已符合就跳過。 |
| `modules/launch_utils.py` | 舊檢查器只辨識 `==` 且接受更高版本；改用共用檢查器。所有 `run_pip install` 都附上主 requirements 與共通 constraints（路徑加引號）。阻止 XPU 使用 CUDA-only 安裝參數。 |
| `extensions/stable-diffusion-webui-tagger-fork/install.py` | 移除每次 uninstall/reinstall ONNX 和無條件安裝 CUDA 11／12 的根因；統一使用目前 venv 的 Python。 |
| `extensions/stable-diffusion-webui-tagger-fork/requirements.txt` | 加上 Hub、ONNX、TensorFlow 相容範圍；移除 WD14 本身不需要的 OpenCV contrib/headless 直接宣告，以及未使用的 graphsurgeon 安裝。 |
| `extensions/stable-diffusion-webui-tagger-fork/tagger/interrogator.py` | 推論時不再臨時安裝 GPU ONNX；Intel 使用 CPU provider；NVIDIA 預載 ORT 官方 extras 的 DLL。 |
| `extensions-builtin/forge_legacy_preprocessors/install.py` | 共用 ONNX 選擇，不再於 NVIDIA 環境補裝 CPU 版；需求字串正確檢查並加引號。 |
| `extensions-builtin/forge_legacy_preprocessors/requirements.txt` | 移除固定 CPU ORT 宣告，ONNX 限定 `>=1.23,<2`。 |
| `extensions/ADetailer-Neo/install.py` | 用 requirements 判斷已安裝版本；符合時不執行安裝。 |
| `extensions/ADetailer-Neo/requirements.txt`（新增） | 將原有 ultralytics 8.3.253、mediapipe 0.10.35 明列供解析與檢查。 |
| `extensions/ADetailer-Neo/lib_adetailer/detection/common.py` | 逐個顯示模型名稱、進度、完成和錯誤；30 秒連線／讀取逾時；檔案鎖、暫存下載、長度檢查及原子替換；錯誤／Ctrl+C 清理暫存，硬中斷殘留於下次清理。已有非空正式模型不重抓。 |
| `extensions/ADetailer-Neo/lib_adetailer/ui.py` | checkpoint、VAE 空清單預設為 None；sampler／scheduler 原本有固定首項，不會為空。 |
| `webui-user-intel-arc.bat` | 保留獨立 venv；固定 `torch 2.14.0+xpu`／`torchvision 0.29.0+xpu`，避免日後首次安裝任意升級。 |
| `webui-user.bat` | 明確使用 CUDA index／原有 CUDA 版本組合，避免繼承先前 Intel 啟動留下的 TORCH_COMMAND。兩個 BAT 都清除 ORT nightly 內部旗標。 |
| `tools/check-installation.py`（新增） | 唯讀驗收：pip check、各 requirements、版本、GPU build／可用性、NVIDIA 套件、ONNX provider。 |
| `tests/test_installation_policy.py`（新增） | CPU 可執行的依賴政策／下載／空模型回歸測試。 |
| `INTEL_ARC_SETUP.md` | 說明沒有 checkpoint 也可啟動 UI，並區分先前使用者實測和本次尚待實機驗收。 |

### 依賴來源

- ONNX 1.23.0 要求 protobuf >=6.31.1；TensorFlow 2.21.0 要求 >=6.31.1,<8；兩者與專案選擇的 `<7` 有交集。
- Gradio 4.40.0 要求 Pillow >=8,<11。`modules/images.py` 會匯入並註冊 HEIF，所以保留 HEIF 功能，改用要求 Pillow >=10.1 的 pillow-heif 0.22.0。
- Transformers 4.57.6 的 Hub 要求為 >=0.34,<1。
- WD14 舊 install.py 同時反覆安裝 CPU／GPU ONNX 並指定 cuDNN cu11/cu12；cuDNN 會再帶入 cublas 等間接依賴。
- 新流程只安裝一種 ORT：Intel 是 `onnxruntime>=1.30,<1.31`；NVIDIA 是 `onnxruntime-gpu[cuda,cudnn]>=1.30,<1.31`，不手工列舊版 CUDA 套件。1.30.0 的官方 extras 已使用 CUDA 13／cuDNN cu13，不能沿用早期 ORT 使用 CUDA 12 的假設。
- MediaPipe 本身仍要求 opencv-contrib-python，而其他上游要求 opencv-python；本次解析兩者皆為 5.0.0.93。沒有用 `--no-deps` 隱藏這些上游需求；兩個套件共用 cv2 命名空間的既有風險仍需實機測試。

來源：[Gradio 4.40.0](https://pypi.org/project/gradio/4.40.0/)、[HEIF 0.22.0 metadata](https://github.com/bigcat88/pillow_heif/blob/v0.22.0/setup.cfg)、[ONNX 1.23.0](https://pypi.org/project/onnx/1.23.0/)、[TensorFlow 2.21.0](https://pypi.org/project/tensorflow/2.21.0/)、[ORT 1.30.0 metadata](https://pypi.org/pypi/onnxruntime-gpu/1.30.0/json)、[ORT CUDA 文件](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

## 已做驗證與限制

- 11 項 CPU 回歸測試通過：版本上限／固定版本／marker、missing package、extras、XPU build 判斷、兩種 ORT 首次及第二次啟動、混裝拒絕、含空格的 constraint 路徑、空 checkpoint／VAE、下載成功快取／殘留清理、截斷後重試、逾時後繼續下一檔、Ctrl+C 清理。部分情境在同一項測試內。
- 用 uv 針對 **Windows x64、Python 3.13** 解析主 requirements、ADetailer、WD14、內建 preprocessor requirements，以及各硬體的 PyTorch／torchvision／ORT。
- XPU 解析 178 個套件，torch 2.14.0+xpu／torchvision 0.29.0+xpu／onnxruntime 1.30.0，沒有 nvidia-* 套件。
- CUDA 解析 165 個套件，torch 2.13.0+cu130／torchvision 0.28.0+cu130／onnxruntime-gpu 1.30.0，保留 NVIDIA 依賴。
- 兩組皆解析為 protobuf 6.33.6、Hub 0.36.2、Pillow 10.4.0、pillow-heif 0.22.0、Gradio 4.40.0、TensorFlow 2.21.0。
- Python 語法與 diff 空白檢查通過（保留既有 CRLF 檔案）。
- 這些是**依賴解析、靜態檢查與模擬回歸測試**。沒有 Windows／B580／RTX 5060 Ti GPU 實機，未宣稱已啟動 UI、實際載入 ONNX CUDA 模型，或完整安裝後 pip check 已通過。
- 主流程以外的可選 InsightFace、Depth Anything 外部 wheel、翻譯服務的動態依賴沒有完成 Windows 安裝驗證。它們仍受主版本 constraints 約束；若不相容會報錯，不會默默改壞共通套件。
- 最初 pip 的「僅 binary、跨平台 dry-run」卡在 filterpy 沒有 binary wheel；之後改用能解析 source metadata 的 uv 完成上述解析。這不代表已驗證 Windows 的 source build。

## B580 全新安裝（Windows CMD）

請使用全新資料夾，保留目前已成功的環境；不要複製舊 venv、設定或模型進來。

```bat
git clone --branch dianaotou-zh-tw --single-branch https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git forge-neo-b580-clean
cd forge-neo-b580-clean
webui-user-intel-arc.bat
```

確認：建立 venv-intel-arc、安裝 XPU PyTorch、ADetailer 顯示模型下載進度，即使 `models\Stable-diffusion` 沒有 checkpoint，仍能到 `http://127.0.0.1:7860`。沒有 checkpoint 時不能進行主模型推論，這是預期行為。

在另一個 CMD 進入該資料夾後執行：

```bat
venv-intel-arc\Scripts\python.exe tools\check-installation.py --backend xpu
```

預期 `No broken requirements found.`、`GPU available: True`、`installed backend: xpu`、`NVIDIA distributions: []`、`RESULT: PASS`。WD14 在 Intel 上使用 ONNX CPU，Forge 主模型使用 XPU。

## RTX 5060 Ti 全新安裝（Windows CMD）

```bat
git clone --branch dianaotou-zh-tw --single-branch https://github.com/DianNaoTou/sd-webui-forge-neo-zh-TW.git forge-neo-nvidia-clean
cd forge-neo-nvidia-clean
webui-user.bat
```

同樣先保持 checkpoint 資料夾空白，確認 UI 啟動及 ADetailer 下載。然後執行：

```bat
venv-nvidia\Scripts\python.exe tools\check-installation.py --backend cuda
```

預期 pip check 無衝突、CUDA build／GPU 可用、只存在 GPU 版 ORT。這台有 nvidia-* 是正常的。後續用一張你已有的圖片執行 WD14，確認實際 ONNX session 可使用 CUDA；provider 列表本身不等於 DLL 載入或推論成功。

## 第二次啟動與中斷驗收

1. 正常關閉 WebUI，再執行相同 BAT。
2. 已符合版本的套件不應重裝，尤其不應看到 onnxruntime uninstall/install 迴圈。
3. 已有 ADetailer 模型不應再出現 Downloading；必要時比較檔案修改時間。
4. 測試中斷下載只能在這個全新測試副本進行：第一次下載時 Ctrl+C，再啟動；應清除 partial 並重抓該未完成模型。可保留 `.lock` 空檔，它不是模型或未完成下載。
5. 額外放入 checkpoint 後確認可載入；本次沒有修改生成演算法。

僅執行 pip check 或查看版本：

```bat
rem B580
venv-intel-arc\Scripts\python.exe -m pip check
venv-intel-arc\Scripts\python.exe -m pip show torch torchvision protobuf Pillow pillow-heif transformers huggingface-hub onnx onnxruntime
rem NVIDIA
venv-nvidia\Scripts\python.exe -m pip check
venv-nvidia\Scripts\python.exe -m pip show torch torchvision protobuf Pillow pillow-heif transformers huggingface-hub onnx onnxruntime-gpu
```

## 副作用與相容邊界

- Pillow／HEIF 採相容舊版，較新版的格式修正不會包含在此組合；待未來升級 Gradio 再整體評估。
- 下載改為逐個進行，輸出更清楚，但首次總下載時間可能較並行長。30 秒是單次網路操作逾時，不是整個模型限時。沒有伺服器 Content-Length 時只能計數，沒有已知 hash 的舊正式檔也不能保證內容正確。
- 已混裝 CPU/GPU ONNX 的舊 venv 會顯示明確錯誤；不做大規模自動卸載。此次驗收請使用全新 venv。
- TensorFlow 固定在 2.21 系列，保留 WD14 原有 TensorFlow／DeepDanbooru 路徑；它依然是較大的下載。
- NVIDIA ONNX 官方 extras 可能另外下載所需 DLL，避免依賴不明來源的 CUDA 11／12；CUDA provider 仍需實機確認。
- XPU PyTorch 固定 2.14.0 配對 torchvision 0.29.0；既有且完整的其他 PyTorch 安裝不會被無條件重装。
- 共通 constraints 會拒絕不相容的第三方擴充依賴，而不是讓它覆蓋核心版本。
- 原有 `--onnxruntime-gpu` nightly 選項保留，未列入本次 stable ORT 驗收；本文件的檢查工具以預設 stable 組合為準。
