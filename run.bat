@echo off
rem Windows equivalent of run.sh: data folder -> features -> predictions.
setlocal
set ROOT=%~dp0
set PYTHONPATH=%ROOT%
set DATA_DIR=%~1
set MODEL_PATH=%~2
set OUTPUT_PATH=%~3
if "%DATA_DIR%"=="" set DATA_DIR=./data
if "%MODEL_PATH%"=="" set MODEL_PATH=./pickle/model.pkl
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=./output/predictions.csv
if "%PYTHON%"=="" set PYTHON=python

if exist "%OUTPUT_PATH%" del "%OUTPUT_PATH%"

"%PYTHON%" "%ROOT%src\generate_features.py" --data-dir "%DATA_DIR%" --out features.parquet
if errorlevel 1 exit /b 1

"%PYTHON%" "%ROOT%src\predict.py" --features features.parquet --model "%MODEL_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

echo Done. Predictions written to %OUTPUT_PATH%
