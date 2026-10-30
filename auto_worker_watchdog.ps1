param(
    [datetime]$Target = [datetime]'2026-06-30T09:00:00',
    [string]$Root = $PSScriptRoot,
    [int]$MaxRestarts = 3
)

$StopFile = Join-Path $Root 'auto_trader.stop'
$LockFile = Join-Path $Root 'auto_session_trader.lock'
$LogFile = Join-Path $Root 'auto_worker_watchdog.log'
$OutLog = Join-Path $Root 'auto_session_live.out.log'
$ErrLog = Join-Path $Root 'auto_session_live.err.log'

function Write-WatchdogLog($Message) {
    "[$((Get-Date).ToString('o'))] $Message" | Add-Content -Path $LogFile -Encoding utf8
}

function Get-LockedWorkerPid {
    if (-not (Test-Path $LockFile)) {
        return 0
    }
    try {
        $lock = Get-Content -Raw -Path $LockFile | ConvertFrom-Json
        return [int]$lock.pid
    } catch {
        return 0
    }
}

function Test-ProcessAlive($PidValue) {
    if ($PidValue -le 0) {
        return $false
    }
    try {
        $null = Get-Process -Id $PidValue -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

Write-WatchdogLog "watchdog started target=$($Target.ToString('o')) root=$Root"
$restartCount = 0

while ((Get-Date) -lt $Target) {
    if (Test-Path $StopFile) {
        Write-WatchdogLog "stop file detected, watchdog exiting"
        exit 0
    }

    $pidValue = Get-LockedWorkerPid
    if (-not (Test-ProcessAlive $pidValue)) {
        if ($restartCount -ge $MaxRestarts) {
            Write-WatchdogLog "max restarts reached, watchdog exiting"
            exit 1
        }
        $restartCount += 1
        Write-WatchdogLog "worker missing, restart #$restartCount"
        Start-Process -FilePath 'python' `
            -ArgumentList @('auto_session_trader.py', '--interval-seconds', '60', '--clear-stop') `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $OutLog `
            -RedirectStandardError $ErrLog `
            -WindowStyle Hidden | Out-Null
    }

    Start-Sleep -Seconds 60
}

Write-WatchdogLog "target reached, watchdog exiting"
