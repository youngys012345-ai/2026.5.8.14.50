# 使用与本项目 requirements 一致的 Python 3.12 虚拟环境启动 Docling Fast（opendataloader-pdf-hybrid）。
# 会从项目根 pipeline.json 读取 hybrid_force_ocr、hybrid_ocr_lang（若存在），默认启用 --force-ocr 以便图片/扫描 PDF 走全页 OCR。
#
# 用法（在项目根执行）:
#   .\scripts\start_docling_hybrid.ps1
#
# 覆盖端口: $env:HYBRID_PORT='5003'; .\scripts\start_docling_hybrid.ps1
# 临时关闭强制 OCR（不推荐图片 PDF）: $env:HYBRID_FORCE_OCR='0'; .\scripts\start_docling_hybrid.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$HybridExe = Join-Path $ProjectRoot ".venv\Scripts\opendataloader-pdf-hybrid.exe"
if (-not (Test-Path $HybridExe)) {
    Write-Error "未找到 $HybridExe。请先执行: py -3.12 -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$Port = if ($env:HYBRID_PORT) { $env:HYBRID_PORT } else { "5002" }

# 默认：图片型 PDF 建议强制 OCR（与 Docling Fast 服务端 DocumentConverter 一致）
$ForceOcr = $true
$OcrLang = "ch_sim,en"

$PipePath = Join-Path $ProjectRoot "pipeline.json"
if (Test-Path $PipePath) {
    try {
        $jsonText = Get-Content -LiteralPath $PipePath -Raw -Encoding UTF8
        $cfg = $jsonText | ConvertFrom-Json
        if ($null -ne $cfg.hybrid_force_ocr) {
            $ForceOcr = [bool]$cfg.hybrid_force_ocr
        }
        if ($cfg.PSObject.Properties.Name -contains "hybrid_ocr_lang" -and $cfg.hybrid_ocr_lang) {
            $OcrLang = [string]$cfg.hybrid_ocr_lang
        }
    }
    catch {
        Write-Warning "读取 pipeline.json 失败，使用默认 force_ocr=$ForceOcr ocr_lang=$OcrLang ： $_"
    }
}

if ($env:HYBRID_FORCE_OCR -eq "0" -or $env:HYBRID_FORCE_OCR -eq "false") {
    $ForceOcr = $false
}
if ($env:HYBRID_OCR_LANG) {
    $OcrLang = $env:HYBRID_OCR_LANG
}

$ArgList = @("--port", $Port)
if ($ForceOcr) {
    $ArgList += "--force-ocr"
}
if ($OcrLang) {
    $ArgList += "--ocr-lang"
    $ArgList += $OcrLang
}

Write-Host "Docling Fast: $HybridExe $($ArgList -join ' ')"
& $HybridExe @ArgList
