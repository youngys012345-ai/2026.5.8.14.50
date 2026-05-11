# 使用与本项目 requirements 一致的 Python 3.12 虚拟环境启动 Docling Fast（opendataloader-pdf-hybrid）。
# 会从项目根 pipeline.json 读取 hybrid_force_ocr、hybrid_ocr_lang、hybrid_ca_bundle（若存在），默认启用 --force-ocr 以便图片/扫描 PDF 走全页 OCR。
#
# 用法（在项目根执行）:
#   .\scripts\start_docling_hybrid.ps1
#
# 覆盖端口: $env:HYBRID_PORT='5003'; .\scripts\start_docling_hybrid.ps1
# 临时关闭强制 OCR（不推荐图片 PDF）: $env:HYBRID_FORCE_OCR='0'; .\scripts\start_docling_hybrid.ps1
#
# ----- 内网 / SSL（EasyOCR 初始化会从 GitHub 等地址下载模型，HTTPS 需信任） -----
# 1) 推荐：向 IT 索取企业根/代理 CA，与 python certifi 包内 cacert.pem 合并为一个 PEM 文件，
#    在 pipeline.json 设置 hybrid_ca_bundle（相对项目根或绝对路径），或启动前设置环境变量 HYBRID_CA_BUNDLE（优先于配置文件）。
#    本脚本会为 hybrid 子进程设置 SSL_CERT_FILE、REQUESTS_CA_BUNDLE、CURL_CA_BUNDLE。
# 2) 离线：在可联网机器上先用同一套 hybrid_ocr_lang 启动过一次 hybrid（完成下载），再将该用户目录下的 .EasyOCR 文件夹
#    复制到内网机同一账户的 %USERPROFILE%\.EasyOCR（路径需一致），可避免运行时下载。

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$HybridExe = Join-Path $ProjectRoot "venv312\Scripts\opendataloader-pdf-hybrid.exe"
if (-not (Test-Path $HybridExe)) {
    Write-Error "未找到 $HybridExe。请先执行: py -3.12 -m venv venv312 ; .\venv312\Scripts\pip install -r requirements.txt"
    exit 1
}

$Port = if ($env:HYBRID_PORT) { $env:HYBRID_PORT } else { "5002" }

# 默认：图片型 PDF 建议强制 OCR（与 Docling Fast 服务端 DocumentConverter 一致）
$ForceOcr = $true
$OcrLang = "ch_sim,en"
$CaBundleFromFile = $null

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
        if ($cfg.PSObject.Properties.Name -contains "hybrid_ca_bundle" -and $cfg.hybrid_ca_bundle) {
            $rel = [string]$cfg.hybrid_ca_bundle
            $CaBundleFromFile = if ([System.IO.Path]::IsPathRooted($rel)) { $rel } else { Join-Path $ProjectRoot $rel }
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

# EasyOCR / urllib / requests 在内网验证 HTTPS 时使用（HYBRID_CA_BUNDLE 优先于 pipeline.json 中的 hybrid_ca_bundle）
$CaBundle = $CaBundleFromFile
if ($env:HYBRID_CA_BUNDLE -and $env:HYBRID_CA_BUNDLE.Trim()) {
    $CaBundle = $env:HYBRID_CA_BUNDLE.Trim()
}
if ($CaBundle) {
    if (Test-Path -LiteralPath $CaBundle) {
        $absCa = (Resolve-Path -LiteralPath $CaBundle).Path
        $env:SSL_CERT_FILE = $absCa
        $env:REQUESTS_CA_BUNDLE = $absCa
        $env:CURL_CA_BUNDLE = $absCa
        Write-Host "Hybrid SSL：已设置 SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE = $absCa"
    }
    else {
        Write-Warning "Hybrid SSL：hybrid_ca_bundle / HYBRID_CA_BUNDLE 指向的路径不存在，跳过 — $CaBundle"
    }
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
