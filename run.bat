@echo off    
set PATH=C:\ProgramData\Anaconda3\Scripts
call %PATH%\activate.bat
cd stock_prediction
python stock_prediction_nse.py