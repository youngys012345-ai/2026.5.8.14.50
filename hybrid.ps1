# OpenDataLoader PDF hybrid 本地服务（Docling）
# 默认端口 5002，与 extract_pdf.py 的 hybrid 模式一致。
# 招标公告多为中文：启用 ch_sim + en；若 PDF 以嵌入文字为主可改为添加 --no-ocr 以提速。
#
# 预留环境变量（可无 GPU 时用 CPU）：
#   OPENDATALOADER_HYBRID_PORT   端口，默认 5002
#   OPENDATALOADER_HYBRID_HOST   监听地址，默认 127.0.0.1
#   OPENDATALOADER_HYBRID_OCR_LANG  如 ch_sim,en
#   OPENDATALOADER_HYBRID_DEVICE    cpu | cuda | mps（按本机 Docling/PyTorch 支持）
#   OPENDATALOADER_HYBRID_EXTRA_ARGS  追加参数（空格分隔，谨慎使用）

$ErrorActionPreference = "Stop"
$port = if ($env:OPENDATALOADER_HYBRID_PORT) { $env:OPENDATALOADER_HYBRID_PORT } else { "5002" }
$hostAddr = if ($env:OPENDATALOADER_HYBRID_HOST) { $env:OPENDATALOADER_HYBRID_HOST } else { "127.0.0.1" }
$ocrLang = if ($env:OPENDATALOADER_HYBRID_OCR_LANG) { $env:OPENDATALOADER_HYBRID_OCR_LANG } else { "ch_sim,en" }
$device = if ($env:OPENDATALOADER_HYBRID_DEVICE) { $env:OPENDATALOADER_HYBRID_DEVICE } else { "cpu" }

Write-Host "启动 opendataloader-pdf-hybrid，端口 $port，设备 $device ..."
Write-Host "健康检查: http://${hostAddr}:${port}/health"
Write-Host "另开终端: python extract_pdf.py <你的.pdf> 或 python extract_pdf.py --config config\pipeline.json <路径>"
Write-Host "客户端可设: `$env:OPENDATALOADER_HYBRID_URL = 'http://127.0.0.1:$port'"
Write-Host ""

$baseArgs = @(
    "--port", $port
    "--host", $hostAddr
    "--ocr-lang", $ocrLang
    "--device", $device
)
if ($env:OPENDATALOADER_HYBRID_EXTRA_ARGS) {
    $extra = $env:OPENDATALOADER_HYBRID_EXTRA_ARGS -split '\s+'
    $baseArgs = $baseArgs + $extra
}
& opendataloader-pdf-hybrid @baseArgs
