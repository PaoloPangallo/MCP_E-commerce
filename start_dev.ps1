$ErrorActionPreference = "Stop"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Avvio Ambiente di Sviluppo (Docker + FastAPI)" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Avvio dei container Docker (Redis, Qdrant, DB)..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERRORE: Docker Compose non è riuscito a partire. Verifica che Docker Desktop sia attivo." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[2/2] Avvio del server FastAPI in modalita' dev..." -ForegroundColor Yellow
uvicorn app.main:app --reload --port 8050

Write-Host "`nServer chiuso. Arresto dei container Docker in corso..." -ForegroundColor Red
docker-compose down
