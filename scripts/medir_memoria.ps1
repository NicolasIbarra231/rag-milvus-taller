# Mide CPU y RAM de cada contenedor con cada LLM cargado y generando (Windows / PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\medir_memoria.ps1
#
# Por modelo: lo carga, toma una lectura en reposo, lanza una pregunta del gold y muestrea
# `docker stats` cada ~2 s mientras genera; se queda con el pico. También guarda la línea de
# `ollama ps` (tamaño en memoria y procesador: CPU o GPU). En CPU no hay VRAM que medir.
# Salida: reports/eval/memoria_por_modelo.csv y reports/eval/memoria_muestras.csv
param(
    [string[]]$Modelos = @("qwen2.5:1.5b", "llama3.2:1b", "llama3.2:3b"),
    [string]$Pregunta = "¿Cómo se llama la especie de planta descubierta en el Cerro Tusa (Venecia, Antioquia), según una noticia de febrero de 2025?"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$salida = "reports\eval\memoria_por_modelo.csv"
$muestrasCsv = "reports\eval\memoria_muestras.csv"
New-Item -ItemType Directory -Force -Path "reports\eval" | Out-Null

function A-MiB([string]$valor) {
    if ($valor -notmatch "^\s*([\d.]+)\s*([KMGT]?i?B)") { return $null }
    $n = [double]$Matches[1]
    switch ($Matches[2]) {
        "B"   { $n / 1MB } "KiB" { $n / 1KB } "kB" { $n / 1000 / 1.048576 }
        "MiB" { $n }     "MB"  { $n / 1.048576 }
        "GiB" { $n * 1024 } "GB" { $n * 1000 / 1.048576 }
        default { $null }
    }
}

# docker stats puede intercalar secuencias de control de la terminal (borrar pantalla, mover
# el cursor) y líneas vacías; se limpian y se descartan las líneas que no tienen los 3 campos.
function Fila-Stats([string]$linea, [string]$hora) {
    $limpia = ($linea -replace '\x1b\[[0-9;?]*[A-Za-z]', '').Trim()
    $p = $limpia -split ";"
    if ($p.Count -lt 3 -or -not $p[0] -or $p[2] -notmatch '^\s*[\d.]+\s*%\s*$') { return $null }
    [pscustomobject]@{ hora = $hora; contenedor = $p[0].Trim();
                       mem_mib = [math]::Round([double](A-MiB (($p[1] -split "/")[0])), 1);
                       cpu_pct = [double]($p[2].Trim().TrimEnd("%")) }
}

function Lectura {
    $hora = (Get-Date).ToUniversalTime().ToString("o")
    foreach ($linea in @(docker stats --no-stream --format "{{.Name}};{{.MemUsage}};{{.CPUPerc}}")) {
        $f = Fila-Stats $linea $hora
        if ($f) { $f }
    }
}

$filas = @()
foreach ($m in $Modelos) {
    Write-Host "`n== $m"
    $pyCargar = "import requests; requests.post('http://localhost:8000/models/load', json={'model': '$m'}, timeout=600)"
    docker compose exec -T api python -c $pyCargar | Out-Null
    Start-Sleep -Seconds 3
    $reposo = @(Lectura)

    $muestreo = Start-Job -ArgumentList (Get-Location).Path -ScriptBlock {
        param($dir)
        Set-Location $dir
        while ($true) {
            docker stats --no-stream --format "{{.Name}};{{.MemUsage}};{{.CPUPerc}}" |
                ForEach-Object { "$((Get-Date).ToUniversalTime().ToString('o'));$_" }
        }
    }
    $pyPreguntar = "import requests; r = requests.post('http://localhost:8000/ask', json={'query': '''$Pregunta''', 'model': '$m'}, timeout=420); print(r.status_code, r.json().get('tiempos', {}).get('total_s'))"
    $resultado = docker compose exec -T api python -c $pyPreguntar
    Stop-Job $muestreo
    $crudas = Receive-Job $muestreo
    Remove-Job $muestreo
    $durante = @(foreach ($linea in @($crudas)) {
        $texto = "$linea"
        $i = $texto.IndexOf(";")
        if ($i -lt 1) { continue }
        $f = Fila-Stats -linea ($texto.Substring($i + 1)) -hora ($texto.Substring(0, $i))
        if ($f) { $f }
    })
    if ($durante.Count -gt 0) {
        $durante | ForEach-Object { $_ | Add-Member -NotePropertyName modelo -NotePropertyValue $m -PassThru } |
            Export-Csv -Append -NoTypeInformation -Encoding UTF8 $muestrasCsv
    } else {
        Write-Warning "No se obtuvieron muestras de docker stats durante la generación de $m."
    }

    $ps = (docker compose exec -T ollama ollama ps | Select-Object -Skip 1 | Where-Object { $_ -match [regex]::Escape($m) }) -join " "
    $pico = { param($c, $campo) ($durante | Where-Object contenedor -eq $c | Measure-Object $campo -Maximum).Maximum }
    $fila = [pscustomobject]@{
        modelo                 = $m
        ollama_reposo_mib      = ($reposo | Where-Object contenedor -eq "rag-ollama").mem_mib
        ollama_pico_mib        = & $pico "rag-ollama" "mem_mib"
        ollama_cpu_pico_pct    = & $pico "rag-ollama" "cpu_pct"
        api_pico_mib           = & $pico "rag-api" "mem_mib"
        milvus_mib             = ($reposo | Where-Object contenedor -eq "rag-milvus").mem_mib
        frontend_mib           = ($reposo | Where-Object contenedor -eq "rag-frontend").mem_mib
        muestras               = @($durante | Where-Object contenedor -eq "rag-ollama").Count
        respuesta              = "$resultado"
        ollama_ps              = ($ps -replace "\s{2,}", " | ")
        fecha_utc              = (Get-Date).ToUniversalTime().ToString("o")
    }
    $fila | Format-List
    $filas += $fila
}
$filas | Export-Csv -NoTypeInformation -Encoding UTF8 $salida
Write-Host "`nMemoria por modelo en $salida (muestras en $muestrasCsv)."
