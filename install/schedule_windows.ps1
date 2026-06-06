# schedule_windows.ps1 — register hourly Risk Oracle watchlist refresh.
# Calls `python -m risk_oracle.cli refresh-watchlist` once per hour.
# Idempotent: re-running unregisters + recreates the task.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$LogDir      = Join-Path $env:LOCALAPPDATA "RiskOracle\Logs"
$Folder      = "\RiskOracle"
$TaskName    = "RiskOracle-WatchlistRefresh"

$Python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $Python) { $Python = (Get-Command py -ErrorAction SilentlyContinue)?.Source }
if (-not $Python) {
    Write-Error "Python not found on PATH. Install Python 3.10+ first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Get-ScheduledTask -TaskName $TaskName -TaskPath "$Folder\" -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath "$Folder\" -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "-m risk_oracle.cli refresh-watchlist" `
    -WorkingDirectory $ProjectRoot

# Hourly trigger, starting 5 minutes from now and repeating forever
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(5)) `
            -RepetitionInterval (New-TimeSpan -Hours 1) `
            -RepetitionDuration ([System.TimeSpan]::MaxValue)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName $TaskName `
    -TaskPath "$Folder\" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Risk Oracle V2.1 — hourly watchlist refresh + dispatch" | Out-Null

Write-Host "OK  Risk Oracle watchlist refresh registered: $Folder\$TaskName"
Write-Host "    Cadence: every hour"
Write-Host "    Notes:   set environment variables (ANTHROPIC_API_KEY, GMAIL_USER, etc.) at"
Write-Host "             user-or-system scope before this task runs — Task Scheduler reads them"
Write-Host "             from the parent environment at task-launch time."
Write-Host "    Inspect: Get-ScheduledTask -TaskPath '$Folder\'"
Write-Host "    Uninstall: powershell -File `"$ProjectRoot\install\uninstall_windows.ps1`""
