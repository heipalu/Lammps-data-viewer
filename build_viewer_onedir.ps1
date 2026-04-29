$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found in PATH."
}

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Install it with: python -m pip install pyinstaller"
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name LammpsDataViewer `
    --collect-all PySide6 `
    --collect-all pyvista `
    --collect-all pyvistaqt `
    --collect-all vtkmodules `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    .\lammps_data_viewer_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
}

Write-Host "Build complete: .\dist\LammpsDataViewer\LammpsDataViewer.exe"
