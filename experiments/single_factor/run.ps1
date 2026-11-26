<#
.SYNOPSIS
  Single entry point for the clean run. Every stage is resumable: completed outputs are skipped.
.EXAMPLE
  .\run.ps1                      # all stages, both file types
  .\run.ps1 -Stage train -Format win64
#>
param(
    [ValidateSet("all", "setup", "download", "test", "prepare", "preprocess", "train", "freeze", "evaluate", "report", "manuscript")]
    [string]$Stage = "all",
    [ValidateSet("both", "dotnet", "win64")]
    [string]$Format = "both"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$formats = if ($Format -eq "both") { @("dotnet", "win64") } else { @($Format) }
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Invoke-Setup {
    if (-not (Test-Path $py)) { Invoke-Checked "python" @("-m", "venv", ".venv") }
    Invoke-Checked $py @("-m", "pip", "install", "-q", "-r", "requirements.txt")
}

function Invoke-Download {
    # Archives of the pinned dataset revision; SHA256 digests are verified by the prepare stage.
    Invoke-Checked $py @("-m", "viewdep.download")
}

function Invoke-PerFormat([string]$name) {
    foreach ($f in $formats) { Invoke-Checked $py @("-m", "viewdep.pipeline", $name, $f) }
}

$order = @("setup", "download", "test", "prepare", "preprocess", "train", "freeze", "evaluate", "report", "manuscript")
$run = if ($Stage -eq "all") { $order } else { @($Stage) }
foreach ($s in $run) {
    Write-Host "== $s"
    switch ($s) {
        "setup"      { Invoke-Setup }
        "download"   { Invoke-Download }
        "test"       { Invoke-Checked $py @("-m", "pytest", "-q") }
        "report"     { Invoke-Checked $py @("-m", "viewdep.report") }
        "manuscript" {
            Push-Location manuscript
            try { Invoke-Checked "latexmk" @("-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex") }
            finally { Pop-Location }
        }
        default      { Invoke-PerFormat $s }
    }
}
