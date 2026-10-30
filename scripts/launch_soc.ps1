[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$FrontendDir = Join-Path $Root 'frontend'
$LogDir = Join-Path $env:LOCALAPPDATA 'CyberThreatLab\launcher-logs'
$ApiBase = 'http://127.0.0.1:8000/api'
$FrontendUrl = 'http://127.0.0.1:5173'
$StreamlitUrl = 'http://127.0.0.1:8502'
$Processes = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

if (-not (Test-Path $Python)) { throw "Project Python environment is missing: $Python" }
if (-not (Test-Path (Join-Path $FrontendDir 'package.json'))) { throw "React frontend is missing: $FrontendDir" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
# Keep the project's interpreter isolated from a parent Hermes PYTHONPATH.
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

function Get-HttpStatus([string]$Url, [int]$TimeoutSec = 2) {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing
        return [int]$response.StatusCode
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            return [int]$_.Exception.Response.StatusCode
        }
        return 0
    }
}

function Test-PortOpen([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        return $task.Wait(500) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-Ready([string]$Name, [string]$Url, [System.Diagnostics.Process]$Process) {
    $until = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $status = Get-HttpStatus $Url
        if ($status -ge 200 -and $status -lt 400) { return }
        if ($Process -and $Process.HasExited) {
            throw "$Name stopped during startup (exit $($Process.ExitCode)). See launcher logs in $LogDir."
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $until)
    throw "$Name did not become ready at $Url within $TimeoutSeconds seconds. See launcher logs in $LogDir."
}

function Start-ServiceProcess([string]$Name, [string]$File, [string[]]$Arguments, [string]$WorkingDirectory) {
    $safeName = $Name.ToLowerInvariant().Replace(' ', '-')
    $stdout = Join-Path $LogDir "$safeName.out.log"
    $stderr = Join-Path $LogDir "$safeName.err.log"
    $process = Start-Process -FilePath $File -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $Processes.Add($process)
    return $process
}

# Reuse healthy services; never kill or silently replace an unknown process.
$apiSourceHash = (Get-FileHash (Join-Path $Root 'api\main.py') -Algorithm SHA256).Hash.ToLowerInvariant()
$flowStoreSourceHash = (Get-FileHash (Join-Path $Root 'cyberlab\flow_store.py') -Algorithm SHA256).Hash.ToLowerInvariant()
$buildMaterial = "$apiSourceHash`:$flowStoreSourceHash"
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try { $ExpectedApiBuildId = ([System.BitConverter]::ToString($sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($buildMaterial)))).Replace('-', '').Substring(0,12).ToLowerInvariant() }
finally { $sha256.Dispose() }
$apiHealth = Get-HttpStatus "$ApiBase/health"
if ($apiHealth -eq 200) {
    $healthBody = Invoke-RestMethod -Uri "$ApiBase/health" -TimeoutSec 3
    if ($healthBody.build_id -ne $ExpectedApiBuildId) {
        throw "A stale API is answering on port 8000 (running build '$($healthBody.build_id)', source build '$ExpectedApiBuildId'). Stop that API manually, then rerun this launcher."
    }
    $flowStatus = Get-HttpStatus "$ApiBase/flows/status"
    if ($flowStatus -eq 404) { throw 'A stale API is answering on port 8000 (health works but /api/flows/status is missing). Stop that API manually, then rerun this launcher.' }
    if ($flowStatus -eq 0) { throw 'The API health route responds, but its full-flow status route is unreachable. Check the API process before continuing.' }
    Write-Host "Reusing API on port 8000 (health HTTP 200; flow status HTTP $flowStatus)."
} else {
    if (Test-PortOpen 8000) { throw 'Port 8000 is occupied by a process that is not the expected SOC API. Stop it manually; this launcher will not terminate unknown processes.' }
    $apiProcess = Start-ServiceProcess 'SOC API' $Python @('-m','uvicorn','api.main:app','--host','127.0.0.1','--port','8000') $Root
    Wait-Ready 'SOC API' "$ApiBase/health" $apiProcess
    $flowStatus = Get-HttpStatus "$ApiBase/flows/status"
    if ($flowStatus -eq 404) { throw 'Started API is stale: /api/flows/status returned 404.' }
    Write-Host "SOC API ready; full-flow index HTTP $flowStatus. (503 means the local full index is missing or inconsistent.)"
}
$statsStatus = Get-HttpStatus "$ApiBase/flows/stats?dataset=expanded" 60
if ($statsStatus -ne 200) { throw "Expanded dataset statistics could not be prepared (HTTP $statsStatus). See API logs in $LogDir." }
Write-Host 'Expanded dataset statistics are ready (cached for fast dashboard loads).'

$frontendStatus = Get-HttpStatus $FrontendUrl
if ($frontendStatus -ge 200 -and $frontendStatus -lt 400) {
    Write-Host "Reusing React frontend at $FrontendUrl (HTTP $frontendStatus)."
} else {
    if (Test-PortOpen 5173) { throw 'Port 5173 is occupied by a process that is not the SOC frontend. Stop it manually; this launcher will not terminate unknown processes.' }
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $frontendProcess = Start-ServiceProcess 'SOC frontend' $npm @('run','dev','--','--host','127.0.0.1') $FrontendDir
    Wait-Ready 'SOC frontend' $FrontendUrl $frontendProcess
    Write-Host "React frontend ready at $FrontendUrl."
}

$streamlitStatus = Get-HttpStatus $StreamlitUrl
if ($streamlitStatus -ge 200 -and $streamlitStatus -lt 400) {
    Write-Host "Reusing Streamlit at $StreamlitUrl (HTTP $streamlitStatus)."
} else {
    if (Test-PortOpen 8502) { throw 'Port 8502 is occupied by a process that is not the expected Streamlit app. Stop it manually; this launcher will not terminate unknown processes.' }
    $streamlitProcess = Start-ServiceProcess 'Streamlit' $Python @('-m','streamlit','run','streamlit_app.py','--server.address','127.0.0.1','--server.port','8502') $Root
    Wait-Ready 'Streamlit' $StreamlitUrl $streamlitProcess
    Write-Host "Streamlit ready at $StreamlitUrl."
}

if (-not $NoBrowser) { Start-Process $StreamlitUrl }
Write-Host 'In Streamlit, choose Workflow > SOC dashboard. The local full-flow SQLite index must be present for full-dataset exploration.'
Write-Host "Launcher logs: $LogDir"
