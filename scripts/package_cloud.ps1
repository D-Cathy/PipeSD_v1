param([string]$Output = "dist/PipeSD-cloud.zip")
$root = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $root $Output
New-Item -ItemType Directory -Force (Split-Path -Parent $outputPath) | Out-Null
Compress-Archive -Force -Path (Join-Path $root "cloud"), (Join-Path $root "shared") -DestinationPath $outputPath
Write-Output $outputPath
