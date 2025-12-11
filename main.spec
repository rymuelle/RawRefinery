# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

datas = [
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas, 
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    name='RawRefinery',
    console=False
)

app = BUNDLE(
    exe,
    icon='Assets/RawRefineryIcon.icns', 
    info_plist={
        'CFBundleName': 'RawRefinery',
        'CFBundleDisplayName': 'RawRefinery',
        'CFBundleExecutable': 'RawRefinery',
        'LSUIElement': False, 
    },
    name='RawRefinery.app', 
    bundle_name='RawRefinery'
)