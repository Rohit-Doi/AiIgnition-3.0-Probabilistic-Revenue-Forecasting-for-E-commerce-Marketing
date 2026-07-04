@echo off
setlocal
set DATA_DIR=%~1
set MODEL_PATH=%~2
set OUTPUT_PATH=%~3
if "%DATA_DIR%"=="" set DATA_DIR=./data
if "%MODEL_PATH%"=="" set MODEL_PATH=./pickle/model.pkl
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=./output/predictions.csv

python src/generate_features.py --data-dir "%DATA_DIR%" --out features.parquet
if errorlevel 1 exit /b 1

python src/predict.py --features features.parquet --model "%MODEL_PATH%" --output "%OUTPUT_PATH%"
if errorlevel 1 exit /b 1

echo Done. Predictions written to %OUTPUT_PATH%
