param(
    [datetime]$Target = [datetime]'2026-06-30T09:00:00',
    [string]$StopFile = (Join-Path $PSScriptRoot 'auto_trader.stop'),
    [string]$LogFile = (Join-Path $PSScriptRoot 'auto_stop_timer.log')
)

"timer started at $((Get-Date).ToString('o')), target=$($Target.ToString('o')), stop=$StopFile" |
    Out-File -FilePath $LogFile -Encoding utf8

while ((Get-Date) -lt $Target) {
    $remaining = [int](($Target - (Get-Date)).TotalSeconds)
    Start-Sleep -Seconds ([Math]::Min(60, [Math]::Max(1, $remaining)))
}

New-Item -ItemType File -Path $StopFile -Force | Out-Null
"stop file created at $((Get-Date).ToString('o'))" |
    Add-Content -Path $LogFile -Encoding utf8
