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
                 'cluster', 'figures', 'report', 'map', 'test', 'clean', 'distclean',
                 'live-day', 'schedule-live', 'unschedule-live')]
    [string]$Target = 'help',

    [int]$Days = 90,

    # Last service date; pinned to the published analysis. 'latest' follows the sources.
    [string]$End = '2026-06-30',

    # Analyse delays without the ridership source.
    [switch]$NoRidership,

    # Optional run name: results go to data\runs\<Run>\ and reports\<Run>\.
    [string]$Run = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = Join-Path $repoRoot 'src'
if ($Run) { $env:MBTA_RUN = $Run }
$python = 'python'
$liveTask = 'MBTA live collector'

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
  live         poll the MBTA V3 API and record live snapshots (10 minutes)
  live-day     poll until 03:00, the end of the service day
  schedule-live   run live-day every day at 05:00 (Windows Task Scheduler)
  unschedule-live remove that scheduled task
  model        train and evaluate the delay and 10+ minute models (Track A)
  model-full   as 'model', plus random forest and KNN
  cluster      cluster stations by reliability and demand (Track B)
  figures      render interactive and static figures
  report       write reports/report.html, reports/story.html and reports/tables/*.csv
  map          export map data and open the 3D map (needs Node.js)
  all          setup + data + model + cluster + figures + report
  test         run the test suite
  clean        remove generated data and figures
  distclean    as 'clean', plus downloaded caches
"@
    }
    'setup' {
        Invoke-Step 'checking the Python version (3.11-3.13)' @('-c', "import sys; v=sys.version_info[:2]; sys.exit(0 if (3,11) <= v <= (3,13) else 'Python 3.11-3.13 required, found %d.%d' % v)")
        Invoke-Step 'installing the exact dependency versions' @('-m', 'pip', 'install', '-r', 'requirements-lock.txt')
    }
    'data' {
        $collect = @('-m', 'mbta_ds.cli', 'collect', '--days', "$Days", '--end', $End) + $(if ($NoRidership) { @('--no-ridership') } else { @() })
        Invoke-Step "collecting $Days days of MBTA data" $collect
        Invoke-Step 'cleaning' @('-m', 'mbta_ds.cli', 'clean', '--refresh')
        Invoke-Step 'extracting features' @('-m', 'mbta_ds.cli', 'features', '--refresh')
        Invoke-Step 'validating data' @('-m', 'mbta_ds.cli', 'validate')
    }
    'live' {
        Invoke-Step 'polling the V3 API' @('-m', 'mbta_ds.cli', 'live', '--minutes', '10', '--interval', '60')
    }
    'live-day' {
        Invoke-Step 'polling the V3 API until 03:00' @('-m', 'mbta_ds.cli', 'live', '--until', '03:00', '--interval', '60')
    }
    'schedule-live' {
        # Runs as the current user, only while logged on: no password is stored.
        # A run missed while the computer was off or asleep starts when it wakes.
        $log = Join-Path $repoRoot 'dataaw3\collector.log'
        New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
        $command = "& '$PSCommandPath' live-day *>> '$log'"
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $repoRoot `
            -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$command`""
        $trigger = New-ScheduledTaskTrigger -Daily -At '05:00'
        $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 23)
        Register-ScheduledTask -TaskName $liveTask -Action $action -Trigger $trigger -Settings $settings `
            -Description 'Records MBTA live predictions, vehicles and alerts (mbta_ds live-day).' -Force | Out-Null
        Write-Host "Scheduled '$liveTask' daily at 05:00. Log: $log. Start it now with: Start-ScheduledTask -TaskName '$liveTask'" -ForegroundColor Green
    }
    'unschedule-live' {
        Unregister-ScheduledTask -TaskName $liveTask -Confirm:$false
        Write-Host "Removed '$liveTask'."
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
        Invoke-Step 'supporting analyses' @('-m', 'mbta_ds.cli', 'extras')
        Invoke-Step 'writing the report' @('-m', 'mbta_ds.cli', 'report')
        Invoke-Step 'writing the plain-language story' @('-m', 'mbta_ds.cli', 'story')
    }
    'map' {
        Invoke-Step 'exporting the network and replay days for the 3D map' @('-m', 'mbta_ds.cli', 'export-map')
        Write-Host '==> starting the map at http://localhost:5173 (Ctrl-C to stop)' -ForegroundColor Cyan
        npm --prefix map install --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
        npm --prefix map run dev
    }
    'test' {
        Invoke-Step 'running tests' @('-m', 'pytest', '-q')
    }
    'all' {
        & $PSCommandPath setup
        & $PSCommandPath data -Days $Days -End $End -NoRidership:$NoRidership
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
