param(
    [string]$TaskName = "Save Zotero arXiv Word Report",
    [string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$OutputDir = "D:\Downloads\lunwen",
    [string]$PythonExe = "python",
    [string]$DailyAt = "20:45"
)

$ErrorActionPreference = "Stop"

if (-not $env:GMAIL_CLIENT_ID -or -not $env:GMAIL_CLIENT_SECRET -or -not $env:GMAIL_REFRESH_TOKEN) {
    Write-Warning "GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN must be set as user environment variables before the scheduled task can run unattended."
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$command = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    "& { Set-Location '$RepoPath'; `$env:PYTHONPATH='src'; & '$PythonExe' -m paper_triage.report_attachment_saver --output-dir '$OutputDir' }"
)

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($command -join " ")
$trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Download the latest Zotero arXiv paper triage .docx attachment from Gmail." `
    -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName'."
Write-Host "Output directory: $OutputDir"
Write-Host "Daily run time: $DailyAt"
