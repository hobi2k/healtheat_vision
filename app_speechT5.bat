@echo off
setlocal EnableDelayedExpansion
chcp 65001
title Demo App Bootstrap (Python 3.11)

cd /d "%~dp0"

echo === STEP 1: Check / Install Python 3.11 ===

where py >nul 2>&1
if errorlevel 1 (
    echo Python not found. Installing Python 3.11...

    set PYTHON_INSTALLER=python-3.11.9-amd64.exe

    if not exist "!PYTHON_INSTALLER!" (
        powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe -OutFile !PYTHON_INSTALLER!"
        if errorlevel 1 (
            echo Failed to download Python installer.
            pause
            exit /b
        )
    )

    "!PYTHON_INSTALLER!" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
)

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python installation failed.
    pause
    exit /b
)

echo === STEP 2: Create venv ===
if not exist ".venv" (
    py -3.11 -m venv .venv
)

echo === STEP 3: Upgrade pip ===
.\.venv\Scripts\python.exe -m pip install --upgrade pip

echo === STEP 4-1: Install CUDA PyTorch ===
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision torchaudio ^
 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo FAILED to install CUDA PyTorch
    pause
    exit /b
)

echo === STEP 4-2: Install other requirements ===
.\.venv\Scripts\python.exe -m pip install -r requirements_diffusion.txt
if errorlevel 1 (
    echo FAILED to install requirements
    pause
    exit /b
)

echo === STEP 5: Verify CUDA ===
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA build:', torch.version.cuda)"

echo === STEP 6: Run app ===
.\.venv\Scripts\python.exe app_speechT5.py

pause