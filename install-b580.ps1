$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot "venv313\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    $runtimeRoot = Join-Path (Split-Path $PSScriptRoot -Parent) "Python313-runtime"
    $runtimePython = Get-ChildItem -Path $runtimeRoot -Filter python.exe -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $runtimePython) {
        throw "Python 3.13 runtime not found under: $runtimeRoot"
    }

    Write-Host "Creating isolated venv313 with: $runtimePython"
    & $runtimePython -m venv (Join-Path $PSScriptRoot "venv313")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv313." }
}

function Invoke-Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $venvPython -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed: $($Arguments -join ' ')"
    }
}

Invoke-Pip install --upgrade pip
Invoke-Pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
Invoke-Pip install --prefer-binary -r requirements.txt

# Bundled extension dependencies that are skipped when WebUI runs with --skip-install.
Invoke-Pip install --prefer-binary jsonschema onnxruntime
Invoke-Pip install --prefer-binary chardet PyExecJS lxml pathos cryptography openai boto3 aliyun-python-sdk-core aliyun-python-sdk-alimt

& $venvPython -c "import torch; assert torch.xpu.is_available(); print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "XPU verification failed."
}

Write-Host "Intel Arc B580 dependencies installed successfully." -ForegroundColor Green
