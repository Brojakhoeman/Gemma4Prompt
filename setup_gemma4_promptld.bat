@echo off
title Gemma4 PromptLD — Auto Setup
color 0A
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║       Gemma4 PromptLD — Auto Setup                                     ║
echo ║       by Brojachoeman                                                  ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ── CONFIG ───────────────────────────────────────────────
set LLAMA_DIR=C:\llama
set MODELS_DIR=C:\models
set LLAMA_EXE=%LLAMA_DIR%\llama-server.exe
set LLAMA_ZIP=%LLAMA_DIR%\llama_install.zip
set LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b8664/llama-b8664-bin-win-cuda-cu12.4-x64.zip
set GGUF_URL=https://huggingface.co/amarck/gemma-4-31b-it-abliterated-GGUF/resolve/main/gemma-4-31b-it-abliterated-t126-Q4_K_M.gguf
set GGUF_FILE=%MODELS_DIR%\gemma-4-31b-it-abliterated-t126-Q4_K_M.gguf
:: ─────────────────────────────────────────────────────────

echo [STEP 1/3] Checking llama-server...
echo.

:: Check if already in PATH
where llama-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ llama-server found in PATH — skipping install.
    goto :check_gguf
)

:: Check C:\llama
if exist "%LLAMA_EXE%" (
    echo ✅ llama-server found at %LLAMA_EXE% — skipping install.
    goto :check_gguf
)

:: Not found — download
echo ⚠  llama-server not found. Downloading to %LLAMA_DIR%...
echo.
echo URL: %LLAMA_URL%
echo.

if not exist "%LLAMA_DIR%" mkdir "%LLAMA_DIR%"

:: Use curl (built into Windows 10+)
curl -L --progress-bar -o "%LLAMA_ZIP%" "%LLAMA_URL%"
if %errorlevel% neq 0 (
    echo.
    echo ❌ Download failed. Check your internet connection.
    echo    You can manually download from:
    echo    %LLAMA_URL%
    echo    and extract to %LLAMA_DIR%
    pause
    exit /b 1
)

echo.
echo Extracting...
powershell -NoProfile -Command "Expand-Archive -Path '%LLAMA_ZIP%' -DestinationPath '%LLAMA_DIR%' -Force"
if %errorlevel% neq 0 (
    echo ❌ Extraction failed.
    pause
    exit /b 1
)

:: Flatten any subfolder — move everything up to C:\llama
for /d %%D in ("%LLAMA_DIR%\*") do (
    echo Flattening %%D...
    move "%%D\*" "%LLAMA_DIR%\" >nul 2>&1
    rmdir "%%D" >nul 2>&1
)

del "%LLAMA_ZIP%" >nul 2>&1

if exist "%LLAMA_EXE%" (
    echo ✅ llama-server installed at %LLAMA_EXE%
) else (
    echo ❌ llama-server.exe not found after extraction.
    echo    Check %LLAMA_DIR% manually.
    pause
    exit /b 1
)

:check_gguf
echo.
echo [STEP 2/3] Checking GGUF model...
echo.

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

:: Check if any GGUF already exists
set GGUF_FOUND=0
for %%F in ("%MODELS_DIR%\*.gguf") do set GGUF_FOUND=1

if %GGUF_FOUND% == 1 (
    echo ✅ GGUF model already present in %MODELS_DIR% — skipping download.
    goto :check_python
)

echo ⚠  No GGUF found in %MODELS_DIR%.
echo.
echo Recommended model: gemma-4-31b-it-abliterated-t126-Q4_K_M.gguf
echo Size: ~17.4 GB
echo.
set /p DOWNLOAD_GGUF="Download it now? (y/n): "
if /i "%DOWNLOAD_GGUF%" == "y" (
    echo.
    echo Downloading GGUF — this will take a while at ~17GB...
    echo You can also download it manually from HuggingFace and drop it in %MODELS_DIR%
    echo.
    curl -L --progress-bar -o "%GGUF_FILE%" "%GGUF_URL%"
    if %errorlevel% neq 0 (
        echo ❌ GGUF download failed. Download manually and place in %MODELS_DIR%
        pause
        exit /b 1
    )
    echo ✅ GGUF downloaded to %GGUF_FILE%
) else (
    echo.
    echo Skipped. Place your GGUF manually in: %MODELS_DIR%
    echo Then restart ComfyUI — it will appear in the node dropdown.
)

:check_python
echo.
echo [STEP 3/3] Checking Python dependencies...
echo.

pip show gradio >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing gradio + requests...
    pip install gradio requests
) else (
    echo ✅ gradio already installed.
)

pip show requests >nul 2>&1
if %errorlevel% neq 0 (
    pip install requests
) else (
    echo ✅ requests already installed.
)

:: Done
echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║   ✅ Setup Complete!                                 ║
echo ╠══════════════════════════════════════════════════════╣
echo ║                                                      ║
echo ║   llama-server : %LLAMA_EXE%
echo ║   Models folder: %MODELS_DIR%                       ║
echo ║                                                      ║
echo ║   Next steps:                                        ║
echo ║   1. Place your GGUF in C:\models\ if not done      ║
echo ║   2. Restart ComfyUI                                 ║
echo ║   3. Add the Gemma4 Prompt Engineer node             ║
echo ║   4. Hit PREVIEW — node handles the rest             ║
echo ║                                                      ║
echo ╚══════════════════════════════════════════════════════╝
echo.
pause
