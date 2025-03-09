@echo off
echo Tạo môi trường ảo cho ứng dụng nhận diện chữ số...

REM Kiểm tra Python đã được cài đặt chưa
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python chưa được cài đặt. Vui lòng cài đặt Python 3.9-3.12 từ https://www.python.org/downloads/
    exit /b 1
)

REM Kiểm tra phiên bản Python
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Phiên bản Python: %PYTHON_VERSION%

REM Kiểm tra virtualenv đã được cài đặt chưa
pip show virtualenv >nul 2>&1
if %errorlevel% neq 0 (
    echo Cài đặt virtualenv...
    pip install virtualenv
)

REM Tạo môi trường ảo nếu chưa tồn tại
if not exist venv (
    echo Tạo môi trường ảo...
    python -m virtualenv venv
) else (
    echo Môi trường ảo đã tồn tại.
)

REM Kích hoạt môi trường ảo và cài đặt các thư viện
echo Kích hoạt môi trường ảo và cài đặt các thư viện...
call venv\Scripts\activate.bat

REM Cài đặt setuptools và wheel trước
echo Cài đặt setuptools và wheel...
pip install setuptools wheel

REM Cài đặt các thư viện từ requirements.txt thay vì setup.py
echo Cài đặt các thư viện từ requirements.txt...
pip install -r requirements.txt

echo.
echo Thiết lập hoàn tất! Để chạy ứng dụng:
echo 1. Kích hoạt môi trường ảo: venv\Scripts\activate
echo 2. Chạy ứng dụng: python run.py
echo.
echo Nhấn phím bất kỳ để thoát...
pause >nul 