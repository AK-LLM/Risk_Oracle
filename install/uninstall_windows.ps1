# uninstall_windows.ps1 — remove Risk Oracle scheduled task.
$ErrorActionPreference = "SilentlyContinue"
$TaskName = "RiskOracle-WatchlistRefresh"
$Folder = "\RiskOracle"
if (Get-ScheduledTask -TaskName $TaskName -TaskPath "$Folder\" -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath "$Folder\" -Confirm:$false
    Write-Host "OK  Removed $Folder\$TaskName"
} else {
    Write-Host "($Folder\$TaskName was not installed)"
}
