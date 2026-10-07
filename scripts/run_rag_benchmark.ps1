<#
.SYNOPSIS
  Measure hybrid passage retrieval against the lexical baseline on today's news.

.DESCRIPTION
  1. Pulls today's headlines once and saves them (SEMANTIC_PASSAGES=false run 1).
  2. Re-scores the SAME headline set with --from-file:
       lexical  x1 more (run 2)
       hybrid   x2 at SEMANTIC_MIN_SIMILARITY 0.30
       hybrid   x1 each at 0.25 and 0.40
  Every run writes docs/benchmarks/<name>.json.

  Explanations are switched off throughout: they cannot change a verdict, so
  they cannot change any number measured here, and leaving them on would only
  add latency and spend free-tier quota.

  Two runs per setting because live search is not deterministic — the same
  query returns different articles minutes apart — and a difference smaller
  than the run-to-run spread is not a difference.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_rag_benchmark.ps1 -Limit 10
#>
param([int]$Limit = 12)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$ml = Join-Path $root "ml-service"
$python = Join-Path $ml ".venv\Scripts\python.exe"
$out = Join-Path $root "docs\benchmarks"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$headlines = Join-Path $out "lexical_run1.json"

$env:EXPLANATIONS_ENABLED = "false"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"   # stream progress instead of buffering a whole run

function Invoke-Run([string]$name, [string]$semantic, [string]$floor, [string[]]$source) {
  $env:SEMANTIC_PASSAGES = $semantic
  $env:SEMANTIC_MIN_SIMILARITY = $floor
  $target = Join-Path $out "$name.json"
  Write-Host "`n=== $name (SEMANTIC_PASSAGES=$semantic, floor=$floor) ===" -ForegroundColor Cyan
  Push-Location $ml
  try { & $python news_benchmark.py @source --save $target } finally { Pop-Location }
}

Invoke-Run "lexical_run1" "false" "0.30" @("--limit", "$Limit")
Invoke-Run "lexical_run2" "false" "0.30" @("--from-file", $headlines)
Invoke-Run "hybrid030_run1" "true" "0.30" @("--from-file", $headlines)
Invoke-Run "hybrid030_run2" "true" "0.30" @("--from-file", $headlines)
Invoke-Run "hybrid025_run1" "true" "0.25" @("--from-file", $headlines)
Invoke-Run "hybrid040_run1" "true" "0.40" @("--from-file", $headlines)

Write-Host "`nDone. Results in $out" -ForegroundColor Green
