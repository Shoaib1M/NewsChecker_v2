<#
.SYNOPSIS
  Start all three NewsChecker services and warm both models before recording.

.DESCRIPTION
  Opens one PowerShell window per service (ML service, Express server, Vite
  client), waits until the ML service reports its NLI model and passage
  embedder ready, then sends one real /api/check so the first claim in the
  recording does not pay for model loading, tokenizer warm-up, or a cold
  Gemini connection.

  Why a script: the first check after a cold start is several times slower
  than every later one, and that is the one an audience sees. Doing it by
  hand before every take is how a step gets skipped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\demo_warmup.ps1
#>

param(
  [string]$WarmupClaim = "NASA launched the Artemis II mission to the Moon",
  [int]$ReadyTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ml = Join-Path $root "ml-service"
$python = Join-Path $ml ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Host "No virtualenv at $python. Create it first:" -ForegroundColor Red
  Write-Host "  cd ml-service; python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
  exit 1
}
if (-not (Test-Path (Join-Path $ml ".env"))) {
  Write-Host "Warning: ml-service\.env not found - explanations will be unavailable (no GOOGLE_API_KEY)." -ForegroundColor Yellow
}

function Start-NewsService([string]$title, [string]$dir, [string]$command) {
  # -NoExit keeps the window open so a crash stays readable during the demo.
  $script = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location -LiteralPath '$dir'; $command"
  Start-Process powershell -ArgumentList @("-NoExit", "-Command", $script) | Out-Null
  Write-Host "Started $title" -ForegroundColor Cyan
}

function Wait-Url([string]$url, [int]$timeout) {
  $deadline = (Get-Date).AddSeconds($timeout)
  while ((Get-Date) -lt $deadline) {
    try { return Invoke-RestMethod -Uri $url -TimeoutSec 5 } catch { Start-Sleep -Seconds 2 }
  }
  throw "Timed out waiting for $url"
}

Start-NewsService "NewsChecker ML service :8000" $ml "& '$python' main.py"
Start-NewsService "NewsChecker server :3001" (Join-Path $root "server") "npm run dev"
Start-NewsService "NewsChecker client :5173" (Join-Path $root "client") "npm run dev"

Write-Host "`nWaiting for the ML service (the first start downloads models)..."
$health = Wait-Url "http://localhost:8000/api/health" $ReadyTimeoutSeconds
Write-Host ("  NLI:              {0} ({1})" -f $health.nli.status, $health.nli.model)
Write-Host ("  Passage ranking:  {0} ({1})" -f $health.passage_ranking.status, $health.passage_ranking.model)
Write-Host ("  Explanations:     {0} ({1})" -f $health.explanations.status, $health.explanations.model)
if ($health.nli.status -ne "ready") {
  Write-Host "NLI is not ready - verdicts will abstain. Check the ML service window." -ForegroundColor Red
}

Write-Host "`nSending one warm-up check: '$WarmupClaim'"
$body = @{ statement = $WarmupClaim } | ConvertTo-Json
$started = Get-Date
try {
  $result = Invoke-RestMethod -Uri "http://localhost:8000/api/check" -Method Post `
    -ContentType "application/json" -Body $body -TimeoutSec 180
  $seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
  Write-Host ("  -> {0} ({1} confidence) in {2}s" -f $result.verification.status, $result.confidence, $seconds) -ForegroundColor Green
} catch {
  Write-Host "  Warm-up check failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nWaiting for the server and client..."
Wait-Url "http://localhost:3001/api/health" 120 | Out-Null
Wait-Url "http://localhost:5173" 120 | Out-Null
Write-Host "`nReady. Open http://localhost:5173 (browser zoom 125%)." -ForegroundColor Green
Write-Host "Before recording: clear the history panel, close .env files, turn notifications off."
