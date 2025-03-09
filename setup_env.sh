#!/bin/bash

echo "Tạo môi trường ảo cho ứng dụng nhận diện chữ số..."

# Kiểm tra Python đã được cài đặt chưa
if ! command -v python3 &> /dev/null; then
    echo "Python chưa được cài đặt. Vui lòng cài đặt Python 3.9-3.12."
    exit 1
fi

# Kiểm tra phiên bản Python
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Phiên bản Python: $PYTHON_VERSION"

# Kiểm tra virtualenv đã được cài đặt chưa
if ! pip3 show virtualenv &> /dev/null; then
    echo "Cài đặt virtualenv..."
    pip3 install virtualenv
fi

# Tạo môi trường ảo nếu chưa tồn tại
if [ ! -d "venv" ]; then
    echo "Tạo môi trường ảo..."
    python3 -m virtualenv venv
else
    echo "Môi trường ảo đã tồn tại."
fi

# Kích hoạt môi trường ảo và cài đặt các thư viện
echo "Kích hoạt môi trường ảo và cài đặt các thư viện..."
source venv/bin/activate

# Cài đặt setuptools và wheel trước
echo "Cài đặt setuptools và wheel..."
pip install setuptools wheel

# Cài đặt các thư viện từ requirements.txt thay vì setup.py
echo "Cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Thiết lập hoàn tất! Để chạy ứng dụng:"
echo "1. Kích hoạt môi trường ảo: source venv/bin/activate"
echo "2. Chạy ứng dụng: python run.py"
echo ""
echo "Nhấn Enter để thoát..."
read 