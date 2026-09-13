$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StreamlitExe = Join-Path $ProjectDir ".venv\Scripts\streamlit.exe"
$AppFile = Join-Path $ProjectDir "app.py"
$OutLog = Join-Path $ProjectDir "streamlit.log"
$ErrLog = Join-Path $ProjectDir "streamlit.err.log"

if (-not (Test-Path $StreamlitExe)) {
    throw "Streamlit executable not found at $StreamlitExe. Install dependencies first."
}

if (-not (Test-Path $AppFile)) {
    throw "Streamlit app not found at $AppFile."
}

$ExistingListener = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($ExistingListener) {
    exit 0
}

Start-Process `
    -FilePath $StreamlitExe `
    -ArgumentList @("run", $AppFile, "--server.port", "8501", "--server.headless", "true") `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden
