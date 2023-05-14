@echo off    
set PATH=C:\Users\%USERNAME%\Anaconda3\Scripts
call %PATH%\activate.bat
cd stock_prediction
python stock_prediction_nse.py
cd ..