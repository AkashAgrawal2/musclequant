# MuscleQuant.spec
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
from pathlib import Path

block_cipher = None

# Collect everything from napari and magicgui
napari_datas, napari_binaries, napari_hidden = collect_all("napari")
magic_datas, magic_binaries, magic_hidden = collect_all("magicgui")

# Optional: adjust to wherever your Cellpose models live when building locally
cellpose_models_path = str(Path.home() / ".cellpose" / "models")

a = Analysis(
    ['musclequant/gui.py'],
    pathex=[],
    binaries=napari_binaries + magic_binaries,
    datas=napari_datas + magic_datas + [
        (cellpose_models_path, "cellpose/models"),
    ],
    hiddenimports=napari_hidden + magic_hidden + [
        "magicgui.backends._qtpy",
        "napari.viewer",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MuscleQuant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='MuscleQuant.app',
    icon=None,
    bundle_identifier=None,
)
