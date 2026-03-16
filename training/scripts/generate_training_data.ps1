# generate_training_data.ps1
# Invokes monti_datagen for each training scene at multiple exposure levels
# to generate EXR input/target pairs for ML denoiser training.
#
# Usage:
#   .\scripts\generate_training_data.ps1 [-MontiDatagen <path>] [-OutputDir <path>]
#                                         [-ScenesDir <path>] [-Width <int>] [-Height <int>]
#                                         [-Spp <int>] [-RefFrames <int>]

param(
    [string]$MontiDatagen = "..\build\app\datagen\Release\monti_datagen.exe",
    [string]$OutputDir = "training_data",
    [string]$ScenesDir = "scenes",
    [int]$Width = 960,
    [int]$Height = 540,
    [int]$Spp = 4,
    [int]$RefFrames = 64
)

$ErrorActionPreference = "Stop"

# Validate resolution is divisible by 4 (required by U-Net 2-level 2x MaxPool)
if ($Width % 4 -ne 0 -or $Height % 4 -ne 0) {
    Write-Error "Resolution ${Width}x${Height} must be divisible by 4"
    exit 1
}

# Validate monti_datagen exists
if (-not (Test-Path $MontiDatagen)) {
    Write-Error "monti_datagen not found: $MontiDatagen"
    Write-Host "Build monti_datagen first, or specify path with -MontiDatagen"
    exit 1
}

$MontiDatagen = Resolve-Path $MontiDatagen

# Scene files to process
$Scenes = @(
    @{ Name = "cornell_box";        File = "cornell_box.glb" },
    @{ Name = "damaged_helmet";     File = "DamagedHelmet.glb" },
    @{ Name = "dragon_attenuation"; File = "DragonAttenuation.glb" }
)

# Exposure levels (EV100)
$Exposures = @(-1.0, -0.5, 0.0, 0.5, 1.0)

# Validate all scene files exist
$MissingScenes = @()
foreach ($Scene in $Scenes) {
    $ScenePath = Join-Path $ScenesDir $Scene.File
    if (-not (Test-Path $ScenePath)) {
        $MissingScenes += $Scene.File
    }
}

if ($MissingScenes.Count -gt 0) {
    Write-Error "Missing scene files in ${ScenesDir}: $($MissingScenes -join ', ')"
    Write-Host "Run these scripts first:"
    Write-Host "  python scripts/export_cornell_box.py --output $ScenesDir/cornell_box.glb"
    Write-Host "  python scripts/download_scenes.py --output $ScenesDir/"
    exit 1
}

# Print configuration
$TotalPairs = $Scenes.Count * $Exposures.Count
$RefSpp = $RefFrames * $Spp
Write-Host "=== Monti Training Data Generation ===" -ForegroundColor Cyan
Write-Host "  monti_datagen:  $MontiDatagen"
Write-Host "  Output:         $OutputDir"
Write-Host "  Scenes:         $ScenesDir"
Write-Host "  Resolution:     ${Width}x${Height}"
Write-Host "  Noisy SPP:      $Spp"
Write-Host "  Reference SPP:  $RefSpp ($RefFrames frames x $Spp)"
Write-Host "  Exposures:      $($Exposures -join ', ') EV"
Write-Host "  Total pairs:    $TotalPairs ($($Scenes.Count) scenes x $($Exposures.Count) exposures)"
Write-Host ""

# Generate data
$PairCount = 0
$StartTime = Get-Date

foreach ($Scene in $Scenes) {
    $ScenePath = Resolve-Path (Join-Path $ScenesDir $Scene.File)

    foreach ($Exposure in $Exposures) {
        $PairCount++
        $ExposureStr = "{0:+0.0;-0.0;0.0}" -f $Exposure
        $OutSubDir = Join-Path $OutputDir "$($Scene.Name)/ev_$($ExposureStr)"
        New-Item -ItemType Directory -Path $OutSubDir -Force | Out-Null

        Write-Host "[$PairCount/$TotalPairs] $($Scene.Name) @ ${ExposureStr} EV" -ForegroundColor Yellow

        & $MontiDatagen `
            --output $OutSubDir `
            --width $Width --height $Height `
            --spp $Spp --ref-frames $RefFrames `
            --exposure $Exposure `
            "$ScenePath"

        if ($LASTEXITCODE -ne 0) {
            Write-Error "monti_datagen failed for $($Scene.Name) @ ${ExposureStr} EV (exit code $LASTEXITCODE)"
            exit 1
        }
    }
}

$Elapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "=== Complete ===" -ForegroundColor Green
Write-Host "  Generated $PairCount EXR pairs in $($Elapsed.ToString('hh\:mm\:ss'))"
Write-Host "  Output: $OutputDir"
