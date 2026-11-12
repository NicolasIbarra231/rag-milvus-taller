# Inicializa la pila de punta a punta (Windows / PowerShell).
#
#   .\scripts\inicializar.ps1            # solo CPU
#   .\scripts\inicializar.ps1 -Gpu       # con GPU NVIDIA
#   .\scripts\inicializar.ps1 -Gpu -Build -Humo
#
# Pasos: levanta los servicios, espera a la API, descarga los LLM del experimento,
# carga los vectores en Milvus si la colección está vacía y verifica la carga.
# Es idempotente: ejecutarlo de nuevo no duplica nada ni vuelve a descargar modelos.
param(
    [switch]$Gpu,     # agrega docker-compose.gpu.yml
    [switch]$Build,   # reconstruye la imagen de la API
    [switch]$Humo     # corre la prueba de humo al final
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$archivos = @("-f", "docker-compose.yml")
if ($Gpu) { $archivos += @("-f", "docker-compose.gpu.yml") }
function Compose { docker compose @archivos @args; if ($LASTEXITCODE -ne 0) { throw "Falló: docker compose $args" } }

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Creado .env a partir de .env.example" }
if (-not (Test-Path "data\encoder_config.json")) { throw "Falta data\encoder_config.json (ver data\README.md)." }

Write-Host "`n[1/5] Levantando servicios..."
if ($Build) { Compose up -d --build } else { Compose up -d }

Write-Host "`n[2/5] Esperando a la API (la primera vez descarga el encoder, ~2 min)..."
$limite = (Get-Date).AddMinutes(10)
do {
    Start-Sleep -Seconds 5
    docker compose @archivos exec -T api curl -fs http://localhost:8000/health/live *> $null
    $lista = ($LASTEXITCODE -eq 0)
    if ((Get-Date) -gt $limite) { throw "La API no respondió en 10 minutos. Revise: docker compose ps y docker compose logs api" }
} until ($lista)
Write-Host "API lista."

Write-Host "`n[3/5] Descargando los LLM del experimento (si ya están, solo se verifican)..."
$modelos = docker compose @archivos exec -T api python -c "import requests; print(' '.join(requests.get('http://localhost:8000/config', timeout=30).json()['modelos_experimento']))"
foreach ($m in ($modelos.Trim() -split '\s+')) {
    Write-Host "  - $m"
    Compose exec -T ollama ollama pull $m
}

Write-Host "`n[4/5] Revisando la colección de Milvus..."
$registros = docker compose @archivos exec -T api python -c "import requests; print(requests.get('http://localhost:8000/health', timeout=60).json()['componentes']['milvus'].get('registros') or 0)"
if ([int]$registros.Trim() -gt 0) {
    Write-Host "La colección ya tiene $($registros.Trim()) registros: no se recarga."
    Compose exec -T api python -m scripts.ingest --verificar
} elseif (Test-Path "data\vectores.parquet") {
    Compose exec -T api python -m scripts.ingest --from-vectors data/vectores.parquet
} else {
    throw "La colección está vacía y no existe data\vectores.parquet. Copie los vectores o use el modo --from-corpus (README, sección 4)."
}

Write-Host "`n[5/5] Listo."
if ($Humo) { Compose exec -T api python -m scripts.smoke_test }
Write-Host "Interfaz: http://localhost:8501   ·   API: http://localhost:8000/docs"
