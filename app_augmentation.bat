@echo off
chcp 65001
title Augmentation App Bootstrap (Python 3.11)

cd /d "%~dp0"

echo === STEP 1: Check Python 3.11 ===
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 not found.
    echo Installing Python 3.11...
    pause
    winget install --id Python.Python.3.11 -e --source winget
    echo Python installed. Restarting launcher...
    pause
    start "" "%~f0"
    exit /b
)

echo === STEP 2: Create venv (Python 3.11) ===
if not exist ".venv" (
    py -3.11 -m venv .venv
)

echo === STEP 3: Upgrade pip (venv) ===
.\.venv\Scripts\python.exe -m pip install --upgrade pip

echo === STEP 4: Install requirements ===
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo FAILED to install requirements
    pause
    exit /b
)

echo === STEP 5: Run app ===
.\.venv\Scripts\python.exe app_augmentation.py

pause
