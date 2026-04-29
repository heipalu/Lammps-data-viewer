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
    --hidden-import pyvista `
    --hidden-import pyvistaqt `
    --hidden-import pyvistaqt.plotting `
    --hidden-import vtkmodules.vtkRenderingCore `
    --hidden-import vtkmodules.vtkRenderingOpenGL2 `
    --hidden-import vtkmodules.vtkRenderingFreeType `
    --hidden-import vtkmodules.vtkInteractionStyle `
    --hidden-import vtkmodules.vtkFiltersSources `
    --hidden-import vtkmodules.vtkFiltersCore `
    --hidden-import vtkmodules.vtkFiltersGeneral `
    --hidden-import vtkmodules.vtkCommonCore `
    --hidden-import vtkmodules.vtkCommonDataModel `
    --hidden-import vtkmodules.vtkCommonExecutionModel `
    --hidden-import vtkmodules.vtkCommonTransforms `
    --hidden-import vtkmodules.vtkIOGeometry `
    --hidden-import vtkmodules.vtkIOLegacy `
    --hidden-import vtkmodules.vtkIOXML `
    --hidden-import vtkmodules.qt.QVTKRenderWindowInteractor `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    --exclude-module PySide6.Qt3DAnimation `
    --exclude-module PySide6.Qt3DCore `
    --exclude-module PySide6.Qt3DExtras `
    --exclude-module PySide6.Qt3DInput `
    --exclude-module PySide6.Qt3DLogic `
    --exclude-module PySide6.Qt3DRender `
    --exclude-module PySide6.QtBluetooth `
    --exclude-module PySide6.QtCharts `
    --exclude-module PySide6.QtDataVisualization `
    --exclude-module PySide6.QtDesigner `
    --exclude-module PySide6.QtGraphs `
    --exclude-module PySide6.QtHelp `
    --exclude-module PySide6.QtLocation `
    --exclude-module PySide6.QtMultimedia `
    --exclude-module PySide6.QtMultimediaWidgets `
    --exclude-module PySide6.QtNetworkAuth `
    --exclude-module PySide6.QtNfc `
    --exclude-module PySide6.QtPdf `
    --exclude-module PySide6.QtPdfWidgets `
    --exclude-module PySide6.QtPositioning `
    --exclude-module PySide6.QtQml `
    --exclude-module PySide6.QtQuick `
    --exclude-module PySide6.QtQuick3D `
    --exclude-module PySide6.QtQuickControls2 `
    --exclude-module PySide6.QtRemoteObjects `
    --exclude-module PySide6.QtScxml `
    --exclude-module PySide6.QtSensors `
    --exclude-module PySide6.QtSerialBus `
    --exclude-module PySide6.QtSerialPort `
    --exclude-module PySide6.QtSpatialAudio `
    --exclude-module PySide6.QtSql `
    --exclude-module PySide6.QtStateMachine `
    --exclude-module PySide6.QtTextToSpeech `
    --exclude-module PySide6.QtUiTools `
    --exclude-module PySide6.QtWebChannel `
    --exclude-module PySide6.QtWebEngineCore `
    --exclude-module PySide6.QtWebEngineQuick `
    --exclude-module PySide6.QtWebEngineWidgets `
    --exclude-module PySide6.QtWebSockets `
    --exclude-module PySide6.QtWebView `
    --exclude-module PySide6.QtXml `
    --exclude-module matplotlib `
    --exclude-module pandas `
    --exclude-module scipy `
    --exclude-module selenium `
    --exclude-module numba `
    --exclude-module llvmlite `
    --exclude-module trame `
    --exclude-module IPython `
    --exclude-module jupyter `
    --exclude-module notebook `
    --exclude-module pytest `
    .\lammps_data_viewer_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE."
}

Write-Host "Build complete: .\dist\LammpsDataViewer\LammpsDataViewer.exe"
