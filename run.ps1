# One entry point (Windows, Docker Desktop with the WSL 2 backend): builds the image if
# needed and runs navdelta in Docker.
#   .\run.ps1                    approach 1 (voxel_diff): full pipeline on .\data -> .\outputs
#   .\run.ps1 scene_graph        approach 2; .\run.ps1 loop_outliers: approach 3; .\run.ps1 rgbd_fusion: approach 4
#                                (an approach name may come first: .\run.ps1 loop_outliers verify)
#   .\run.ps1 verify             run twice, require identical results
#   .\run.ps1 check-data         check the recordings
#   .\run.ps1 run --cpu          force the CPU
# Folders: $env:NAVDELTA_DATA_DIR (default .\data), $env:NAVDELTA_OUT_DIR (.\outputs),
#          $env:NAVDELTA_MODEL_DIR (.\.hf_cache)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
foreach ($folder in @($(if ($env:NAVDELTA_OUT_DIR) { $env:NAVDELTA_OUT_DIR } else { '.\outputs' }),
                      $(if ($env:NAVDELTA_MODEL_DIR) { $env:NAVDELTA_MODEL_DIR } else { '.\.hf_cache' }))) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

[string[]]$rest = @($args)
$approach = 'voxel_diff'
if ($rest.Count -gt 0 -and @('voxel_diff', 'scene_graph', 'loop_outliers', 'rgbd_fusion') -contains $rest[0]) {
    $approach = $rest[0]; $rest = @($rest | Select-Object -Skip 1)
}
$service = "$approach-cpu"
$hasGpu = $false
try { nvidia-smi *> $null; $hasGpu = ($LASTEXITCODE -eq 0) } catch { $hasGpu = $false }
if ($hasGpu -and -not ($rest -contains '--cpu')) { $service = $approach }

docker compose build $service
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed ($LASTEXITCODE)" }
[string[]]$navArgs = @($rest)
if ($navArgs.Count -eq 0) { $navArgs = @('run') }
docker compose run --rm $service @navArgs
exit $LASTEXITCODE
