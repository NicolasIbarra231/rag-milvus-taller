# Corre todo lo que faltaba para el informe, en orden y de forma reanudable (Windows).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\ejecutar_pendientes.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\ejecutar_pendientes.ps1 -Solo robustez,memoria
#
# Cada paso deja una marca en reports\.estado\<paso>.ok al terminar bien; si se interrumpe,
# volver a ejecutar el script salta lo ya hecho. Todo lo que imprime queda además en
# reports\ejecucion_pendientes.log. Tiempo total estimado en CPU: 2,5 a 3 horas.
#
#   entorno           hardware y versiones (reports\entorno\)
#   corpus            descarga dragons_rag con open_corpus_co_es a data\documentos_rag.parquet
#   base_corpus       struct256o32 + e5-small reconstruida desde el corpus (control de reproducibilidad)
#   chunk128          ablación de tamaño de chunk: 128 tokens, solapamiento 16
#   chunk512          ablación de tamaño de chunk: 512 tokens, solapamiento 64
#   bge_m3            ablación de encoder: BAAI/bge-m3 (otra colección)
#   muestra10k        muestra de 10.000 documentos (otra colección)
#   gen_convencional  generación con qwen2.5:1.5b y contexto convencional
#   gen_k3            generación con qwen2.5:1.5b y k = 3
#   robustez          inyección en documentos y fuentes contradictorias (3 modelos)
#   memoria           RAM y CPU por modelo durante la generación
#   resumen           tablas de ablaciones y gráficas
param([string[]]$Solo = @())
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path "reports\.estado", "reports\ablaciones" | Out-Null
Start-Transcript -Append -Path "reports\ejecucion_pendientes.log" | Out-Null

$TOKENIZER = "intfloat/multilingual-e5-small"   # mismos cortes de chunk para todos los encoders
$CORPUS = "data/documentos_rag.parquet"

function Paso([string]$nombre, [scriptblock]$accion) {
    $marca = "reports\.estado\$nombre.ok"
    if ($Solo.Count -gt 0 -and $Solo -notcontains $nombre) { return }
    if (Test-Path $marca) { Write-Host "`n[$nombre] ya hecho (borre $marca para repetirlo)."; return }
    Write-Host "`n==================== [$nombre] $(Get-Date -Format 'HH:mm:ss') ===================="
    $t0 = Get-Date
    & $accion
    if ($LASTEXITCODE -ne 0) { throw "El paso '$nombre' falló (código $LASTEXITCODE). Corrija y vuelva a ejecutar el script." }
    Set-Content -Path $marca -Value ((Get-Date).ToString("o"))
    Write-Host ("[{0}] listo en {1:N1} min" -f $nombre, ((Get-Date) - $t0).TotalMinutes)
}

function Ablacion([string]$etiqueta, [string]$coleccion, [string]$encoder, [int]$muestra, [int]$w, [int]$o) {
    if (-not (Test-Path $CORPUS.Replace("/", "\"))) { throw "Falta ${CORPUS}: corra primero el paso 'corpus'." }
    $vars = @("-e", "MILVUS_COLLECTION=$coleccion", "-e", "ENCODER_CONFIG=$encoder")
    # Los vectores se guardan al terminar de codificar: si la carga a Milvus falla después,
    # el reintento los reutiliza en vez de volver a codificar (la muestra de 10.000 tarda ~45 min).
    $cache = "reports/ablaciones/vectores_$etiqueta.parquet"
    if (Test-Path $cache.Replace("/", "\")) {
        Write-Host "Reutilizando los vectores ya calculados en $cache"
        docker compose exec -T @vars api python -m scripts.ingest --from-vectors $cache
    } else {
        docker compose exec -T @vars api python -m scripts.ingest --from-corpus $CORPUS --muestra $muestra `
            --max-tokens $w --overlap $o --tokenizer $TOKENIZER --guardar-vectores $cache
    }
    if ($LASTEXITCODE -ne 0) { return }
    docker compose exec -T @vars api python -m scripts.evaluar_gold recuperacion --local --k-max 8 `
        --etiqueta $etiqueta --salida reports/ablaciones
}

# Comprobaciones previas
docker compose exec -T api curl -fs http://localhost:8000/health/live | Out-Null
if ($LASTEXITCODE -ne 0) { throw "La API no responde. Levante la pila (docker compose ... up -d) y espere a que esté healthy." }
$memApi = (Get-Content .env | Where-Object { $_ -match "^API_MEM=" }) -replace "API_MEM=", ""
if ($memApi -match "^(\d+)g$" -and [int]$Matches[1] -lt 6) {
    Write-Warning "API_MEM=$memApi en .env. Para bge-m3 y la muestra de 10.000 se recomiendan API_MEM=8g y API_CPUS=6 (y recrear la API)."
}

Paso "entorno" { powershell -ExecutionPolicy Bypass -File .\scripts\registrar_entorno.ps1 }

$SHA_CORPUS = "F4A669F17EAEDD35"   # inicio del SHA-256 publicado de dragons_rag_2026_10_19
function Corpus-Valido {
    if (-not (Test-Path "data\documentos_rag.parquet")) { return $false }
    (Get-FileHash "data\documentos_rag.parquet" -Algorithm SHA256).Hash.StartsWith($SHA_CORPUS)
}

Paso "corpus" {
    if (Corpus-Valido) { Write-Host "Ya existe data\documentos_rag.parquet y su SHA-256 coincide."; $global:LASTEXITCODE = 0; return }
    Remove-Item "data\documentos_rag.parquet" -ErrorAction SilentlyContinue   # copia incompleta de un intento anterior
    docker compose exec -T api sh -c "test -s /tmp/documentos_rag.parquet || python -m scripts.descargar_corpus --salida /tmp/documentos_rag.parquet"
    if ($LASTEXITCODE -eq 0) { docker cp rag-api:/tmp/documentos_rag.parquet data\documentos_rag.parquet }
    if ($LASTEXITCODE -eq 0 -and -not (Corpus-Valido)) {
        Remove-Item "data\documentos_rag.parquet" -ErrorAction SilentlyContinue
        throw "La copia de data\documentos_rag.parquet quedó incompleta (¿falta espacio en el disco?)."
    }
}

Paso "base_corpus" { Ablacion "base_corpus" "rag_struct256_e5_corpus" "data/encoder_config.json" 500 256 32 }
Paso "chunk128"    { Ablacion "chunk128"    "rag_struct128_e5"        "data/encoder_config.json" 500 128 16 }
Paso "chunk512"    { Ablacion "chunk512"    "rag_struct512_e5"        "data/encoder_config.json" 500 512 64 }
Paso "bge_m3"      { Ablacion "bge_m3"      "rag_struct256_bgem3"     "configs/encoders/bge-m3.json" 500 256 32 }
Paso "muestra10k"  { Ablacion "muestra10k"  "rag_struct256_e5_10k"    "data/encoder_config.json" 10000 256 32 }

Paso "gen_convencional" {
    docker compose exec -T api python -m scripts.evaluar_gold generacion --modelos qwen2.5:1.5b `
        --estrategia convencional --etiqueta ctx_convencional
    if ($LASTEXITCODE -eq 0) { docker compose exec -T api python -m scripts.evaluar_gold planilla-ablacion --etiqueta ctx_convencional }
    if ($LASTEXITCODE -eq 0) { docker compose exec -T api python -m scripts.evaluar_gold resumen --etiqueta ctx_convencional }
}
Paso "gen_k3" {
    docker compose exec -T api python -m scripts.evaluar_gold generacion --modelos qwen2.5:1.5b --k 3 --etiqueta k3
    if ($LASTEXITCODE -eq 0) { docker compose exec -T api python -m scripts.evaluar_gold planilla-ablacion --etiqueta k3 }
    if ($LASTEXITCODE -eq 0) { docker compose exec -T api python -m scripts.evaluar_gold resumen --etiqueta k3 }
}

Paso "robustez" { docker compose exec -T api python -m scripts.robustez }
Paso "memoria"  { powershell -ExecutionPolicy Bypass -File .\scripts\medir_memoria.ps1 }

Paso "resumen" {
    docker compose exec -T api python -m scripts.evaluar_gold resumen
    docker compose exec -T api pip install -q matplotlib==3.9.2
    docker compose exec -T api python -m scripts.graficas_eval
    docker compose exec -T api python -m scripts.resumen_ablaciones
}

Write-Host "`nTodo listo. Falta solo calificar las respuestas nuevas de las ablaciones de generación:"
Get-ChildItem reports\eval\planilla_ctx_convencional.xlsx, reports\eval\planilla_k3.xlsx -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host "Después: docker compose exec api python -m scripts.evaluar_gold calificaciones-ablacion --etiqueta <ctx_convencional|k3>"
Stop-Transcript | Out-Null
