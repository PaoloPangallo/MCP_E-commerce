@echo off
echo ===================================================
echo   Avvio Ambiente di Sviluppo (Docker + FastAPI)
echo ===================================================

echo.
echo [1/2] Avvio dei container Docker (Redis, Qdrant, DB)...
docker-compose up -d

echo.
echo [2/2] Avvio del server FastAPI in modalita' dev...
uvicorn app.main:app --reload

echo.
echo Chiusura dell'applicazione... Appena lo interrompi, vuoi spegnere i container? (Premi CTRL+C per non spegnerli)
pause
docker-compose down
