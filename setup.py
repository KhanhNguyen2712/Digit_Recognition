from setuptools import find_packages, setup

setup(
    name="digit_recognition",
    version="0.1.0",
    description="Ứng dụng nhận diện chữ số viết tay sử dụng Deep Learning",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.16.1",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
    ],
    python_requires=">=3.9,<3.13",  # TensorFlow mới nhất yêu cầu Python 3.9-3.12
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
