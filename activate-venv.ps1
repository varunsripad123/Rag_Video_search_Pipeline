# Activate-Venv.ps1
# Production-ready script to activate Python venv in PowerShell with checks

param(
    [string]$VenvPath = ".\venv"  # Default venv folder relative to current directory
)

# Resolve full path for the venv activation script
$activateScript = Join-Path -Path $VenvPath -ChildPath "Scripts\Activate.ps1"

if (-Not (Test-Path $activateScript)) {
    Write-Error "Cannot find Activate.ps1 script at path: $activateScript"
    exit 1
}

try {
    Write-Host "Activating virtual environment from: $activateScript"
    # Use Dot-source to run the Activate.ps1 script in current scope
    . $activateScript

    # Confirm activation by checking $env:VIRTUAL_ENV environment variable
    if (-Not $env:VIRTUAL_ENV) {
        Write-Error "Failed to activate virtual environment."
        exit 1
    }
    else {
        Write-Host "Virtual environment activated successfully."
        Write-Host "Python executable: $(Get-Command python | Select-Object -ExpandProperty Source)"
    }
}
catch {
    Write-Error "Exception during activation: $_"
    exit 1
}
