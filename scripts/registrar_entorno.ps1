# Registra el hardware y las versiones del equipo anfitrión (Windows) en reports/entorno/.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\registrar_entorno.ps1
#
# Docker no ve el hardware real del equipo; este script lo toma de Windows y lo junta con
# las versiones de Docker, WSL, Ollama y los metadatos de cada modelo (cuantización,
# parámetros, digest), como la celda de entorno del notebook (Parte 5).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$dir = "reports\entorno"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

function Texto($bloque) { try { (& $bloque 2>&1 | Out-String).Replace("`0", "").Trim() } catch { $null } }

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
$equipo = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory
$so = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
$gpus = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
$wslconfig = Join-Path $env:USERPROFILE ".wslconfig"

$host_info = [ordered]@{
    fecha_utc        = (Get-Date).ToUniversalTime().ToString("o")
    cpu              = $cpu.Name
    nucleos          = $cpu.NumberOfCores
    hilos            = $cpu.NumberOfLogicalProcessors
    frecuencia_mhz   = $cpu.MaxClockSpeed
    ram_total_gb     = [math]::Round($equipo.TotalPhysicalMemory / 1GB, 1)
    equipo           = "$($equipo.Manufacturer) $($equipo.Model)"
    sistema          = "$($so.Caption) $($so.Version) (build $($so.BuildNumber))"
    gpus             = @($gpus | ForEach-Object { "$($_.Name) · driver $($_.DriverVersion)" })
    nvidia_smi       = Texto { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader }
    docker           = Texto { docker version --format "cliente {{.Client.Version}} · servidor {{.Server.Version}}" }
    docker_compose   = Texto { docker compose version --short }
    docker_memoria_vm_gb = Texto { docker info --format "{{.MemTotal}}" }
    docker_cpus_vm   = Texto { docker info --format "{{.NCPU}}" }
    wsl              = Texto { wsl --version }
    wslconfig        = if (Test-Path $wslconfig) { Get-Content $wslconfig -Raw } else { "(no existe: valores por defecto de WSL)" }
    imagenes         = Texto { docker compose images }
    limites_env      = (Get-Content .env | Where-Object { $_ -match "^(MILVUS|OLLAMA|API|FRONTEND)_(CPUS|MEM)=" })
}
if ($host_info.docker_memoria_vm_gb -match "^\d+$") {
    $host_info.docker_memoria_vm_gb = [math]::Round([double]$host_info.docker_memoria_vm_gb / 1GB, 1)
}
$host_info | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 "$dir\entorno_host.json"

# Metadatos de cada modelo desde Ollama (/api/show), igual que la celda 183 del notebook.
$py = @"
import json, requests
from app.config import settings
from app import entorno
base = settings().ollama_url
salida = {'ollama_version': requests.get(base + '/api/version', timeout=10).json().get('version'),
          'proceso_api': entorno.proceso(), 'modelos': {}}
for m in settings().modelos_experimento:
    d = requests.post(base + '/api/show', json={'model': m}, timeout=30).json()
    salida['modelos'][m] = {'details': d.get('details'), 'parameters': d.get('parameters'),
                            'digest': entorno.modelo(m).get('digest')}
print(json.dumps(salida, ensure_ascii=False, indent=2))
"@
docker compose exec -T api python -c $py | Set-Content -Encoding UTF8 "$dir\ollama_modelos.json"

Write-Host "Entorno registrado en $dir\entorno_host.json y $dir\ollama_modelos.json"
Write-Host ("CPU: {0} · {1} hilos · RAM {2} GB" -f $host_info.cpu, $host_info.hilos, $host_info.ram_total_gb)
