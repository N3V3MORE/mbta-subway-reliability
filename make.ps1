<#
.SYNOPSIS
    Windows equivalent of the `makefile`, for machines without `make` installed.

.DESCRIPTION
    The repository ships a `makefile` because the course rubric requires one, but
    `make` is not available on a stock Windows box. This script exposes the same
    targets so the pipeline is runnable either way.

.EXAMPLE
    .\make.ps1 all
    .\make.ps1 data -Days 180
    .\make.ps1 test
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'all', 'setup', 'data', 'live', 'model', 'model-full',
                 'cluster', 'figures', 'report', 'test', 'clean', 'distclean')]
    [string]$Target = 'help',

    [int]$Days = 90,

    # Optional last service date, e.g. -End 2026-02-28.
    [string]$End = '',

    # Optional run name: results go to data\runs\<Run>\ and reports\<Run>\.
    [string]$Run = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = Join-Path $repoRoot 'src'
if ($Run) { $env:MBTA_RUN = $Run }
$python = 'python'

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "step failed with exit code $LASTEXITCODE`: $Label"
    }
}

function Remove-Paths {
    param([string[]]$Paths)
    foreach ($path in $Paths) {
        $full = Join-Path $repoRoot $path
        if (Test-Path $full) {
            Remove-Item -Recurse -Force $full
            Write-Host "removed $path"
        }
    }
}

switch ($Target) {
    'help' {
        Write-Host @"
Targets:
  setup        install Python dependencies
  data         download, clean, build features, validate (Days=$Days)
  live         poll the MBTA V3 API and record live snapshots
  model        train and evaluate the delay and 10+ minute models (Track A)
  model-full   as 'model', plus random forest and KNN
  cluster      cluster stations by reliability and demand (Track B)
  figures      render interactive and static figures
  report       write reports/report.html, reports/story.html and reports/tables/*.csv
  all          setup + data + model + cluster + figures + report
  test         run the test suite
  clean        remove generated data and figures
  distclean    as 'clean', plus downloaded caches
"@
    }
    'setup' {
        Invoke-Step 'installing dependencies' @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-Step 'installing requirements' @('-m', 'pip', 'install', '-r', 'requirements.txt')
    }
    'data' {
        $collect = @('-m', 'mbta_ds.cli', 'collect', '--days', "$Days") + $(if ($End) { @('--end', $End) } else { @() })
        Invoke-Step "collecting $Days days of MBTA data" $collect
        Invoke-Step 'cleaning' @('-m', 'mbta_ds.cli', 'clean', '--refresh')
        Invoke-Step 'extracting features' @('-m', 'mbta_ds.cli', 'features', '--refresh')
        Invoke-Step 'validating data' @('-m', 'mbta_ds.cli', 'validate')
    }
    'live' {
        Invoke-Step 'polling the V3 API' @('-m', 'mbta_ds.cli', 'live', '--minutes', '10', '--interval', '60')
    }
    'model' {
        Invoke-Step 'training and evaluating delay models' @('-m', 'mbta_ds.cli', 'train')
        Invoke-Step 'early warning and prediction ranges' @('-m', 'mbta_ds.cli', 'tail')
    }
    'model-full' {
        Invoke-Step 'training with the costly model set' @('-m', 'mbta_ds.cli', 'train', '--full')
        Invoke-Step 'early warning and prediction ranges' @('-m', 'mbta_ds.cli', 'tail')
    }
    'cluster' {
        Invoke-Step 'clustering stations' @('-m', 'mbta_ds.cli', 'cluster')
    }
    'figures' {
        Invoke-Step 'rendering figures' @('-m', 'mbta_ds.cli', 'figures')
    }
    'report' {
        Invoke-Step 'writing the report' @('-m', 'mbta_ds.cli', 'report')
        Invoke-Step 'writing the plain-language story' @('-m', 'mbta_ds.cli', 'story')
    }
    'test' {
        Invoke-Step 'running tests' @('-m', 'pytest', '-q')
    }
    'all' {
        & $PSCommandPath setup
        & $PSCommandPath data -Days $Days -End $End
        & $PSCommandPath model
        & $PSCommandPath cluster
        & $PSCommandPath figures
        & $PSCommandPath report
        Write-Host ''
        Write-Host 'Pipeline complete. Open reports/report.html.' -ForegroundColor Green
    }
    'clean' {
        Remove-Paths @('data\processed', 'reports')
    }
    'distclean' {
        Remove-Paths @('data\processed', 'reports', 'data\raw', 'data\analysis_window.json')
    }
}
