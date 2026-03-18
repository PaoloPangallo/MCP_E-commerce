param(
    [int]$Port = 8050,
    [string]$StartCommand = "python -m uvicorn app.main:app --reload --port 8050"
)

Write-Host "Controllo processi sulla porta $Port..." -ForegroundColor Cyan

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($procId in $pids) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            Write-Host "Trovato processo PID $procId ($($proc.ProcessName)) - lo termino..." -ForegroundColor Yellow

            Stop-Process -Id $procId -Force
            Write-Host "Processo $procId terminato" -ForegroundColor Green
        }
        catch {
            Write-Host "Errore nel terminare PID $procId" -ForegroundColor Red
        }
    }
} else {
    Write-Host "Nessun processo sulla porta $Port" -ForegroundColor Green
}

Start-Sleep -Seconds 1

Write-Host "Avvio server..." -ForegroundColor Cyan

Invoke-Expression $StartCommand