# Windows PowerShell version of scripts/run_full_suite.sh (paper setting, results -> results\<exp>_full*\)
# Usage (from the repo folder):
#   powershell -ExecutionPolicy Bypass -File scripts\run_full_suite.ps1 -Workers 8
param([int]$Workers = 4)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PY) { $env:PY } else { "python" }
function Invoke-Exp([string[]]$a) {
    Write-Host ">> $py run.py $($a -join ' ')"
    & $py run.py @a
    if ($LASTEXITCODE -ne 0) { throw "failed: run.py $($a -join ' ')" }
}
$w = @("--workers", "$Workers")
Invoke-Exp (@("--exp", "repro_cv96", "--protocol", "full") + $w)                  # E1 paper models
Invoke-Exp (@("--exp", "p2_norm", "--protocol", "full") + $w)                     # E2 peak normalisation
Invoke-Exp (@("--exp", "p2_zscore", "--protocol", "full") + $w)                   # E2 z-score
Invoke-Exp (@("--exp", "p3_ensemble", "--protocol", "full") + $w)                 # E3 5-seed ensembles
Invoke-Exp (@("--exp", "p2_norm", "--protocol", "full", "--input", "goes_b") + $w) # E4 GOES XRS-B input
Invoke-Exp (@("--exp", "p2_norm", "--protocol", "full", "--input", "fism") + $w)   # E4 FISM2 0.1-0.8 nm input
Invoke-Exp (@("--exp", "p2_norm", "--protocol", "full", "--exclude-near-test-hours", "24") + $w)  # E5 leakage
Write-Host "done - summaries: results\*_full*\summary.csv"
