# run_coverage.ps1 — Build Debug and run OpenCppCoverage on monti_tests
# Produces: coverage_report/ (HTML) and coverage.xml (Cobertura) at repo root.
param(
    [string]$BuildDir = "build",
    [string]$Config = "Debug"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

# --- Locate OpenCppCoverage ---
$OccExe = (Get-Command OpenCppCoverage -ErrorAction SilentlyContinue).Source
if (-not $OccExe) {
    # Check default install location
    $DefaultPath = "C:\Program Files\OpenCppCoverage\OpenCppCoverage.exe"
    if (Test-Path $DefaultPath) {
        $OccExe = $DefaultPath
    } else {
        Write-Error "OpenCppCoverage not found on PATH or in default location. Install from https://github.com/OpenCppCoverage/OpenCppCoverage/releases"
        exit 1
    }
}
Write-Host "Using: $OccExe" -ForegroundColor Gray

# --- Build Debug configuration ---
Write-Host "=== Building $Config configuration ===" -ForegroundColor Cyan
cmake --build "$RepoRoot/$BuildDir" --config $Config --target monti_tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit 1
}

# --- Locate test executable ---
$TestExe = "$RepoRoot/$BuildDir/$Config/monti_tests.exe"
if (-not (Test-Path $TestExe)) {
    Write-Error "Test executable not found at $TestExe"
    exit 1
}

# --- Source directories to include in coverage ---
$SourceDirs = @(
    "$RepoRoot\renderer"
    "$RepoRoot\scene"
    "$RepoRoot\denoise"
    "$RepoRoot\capture"
    "$RepoRoot\app"
)

# --- Build OpenCppCoverage arguments ---
$occArgs = @()
foreach ($dir in $SourceDirs) {
    $occArgs += "--sources", $dir
}

# Exclude test code, build artifacts, and third-party/fetched dependencies
$occArgs += "--excluded_sources", "$RepoRoot\tests"
$occArgs += "--excluded_sources", "$RepoRoot\build"
$occArgs += "--excluded_sources", "$RepoRoot\external"
$occArgs += "--excluded_sources", "$RepoRoot\deps"
$occArgs += "--excluded_sources", "$RepoRoot\_deps"

# HTML report
$HtmlDir = "$RepoRoot\coverage"
$occArgs += "--export_type", "html:$HtmlDir"

# Cobertura XML report
$XmlFile = "$RepoRoot\coverage.xml"
$occArgs += "--export_type", "cobertura:$XmlFile"

# The test executable and its arguments
$occArgs += "--", $TestExe

Write-Host "=== Running OpenCppCoverage ===" -ForegroundColor Cyan
& $OccExe @occArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "OpenCppCoverage failed."
    exit 1
}

Write-Host ""
Write-Host "=== Coverage complete ===" -ForegroundColor Green
Write-Host "  HTML report : $HtmlDir\index.html"
Write-Host "  Cobertura   : $XmlFile"
