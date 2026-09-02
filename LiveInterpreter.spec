# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Live Interpreter (Windows, onedir, windowed).

import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# customtkinter ships .json theme data + fonts that must be bundled.
ctk_datas = collect_data_files("customtkinter")

a = Analysis(
    ["stream_gui.py"],
    pathex=["D:\\new dir"],
    binaries=[],
    datas=ctk_datas,
    hiddenimports=[
        "customtkinter",
        "sounddevice",
        "soundfile",
        "pyaudiowpatch",
        "websocket",
        "groq",
        "dotenv",
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter.test", "gradio", "discord", "matplotlib", "numpy.tests",
        "pytest", "setuptools", "pip", "wheel",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveInterpreter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # windowed app — no console flash
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="LiveInterpreter",
)
