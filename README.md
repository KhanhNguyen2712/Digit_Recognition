# Handwritten Digit Recognition Application

This application uses Deep Learning to recognize handwritten digits (0-9) drawn with a mouse.

## Features

- Simple interface for drawing digits with a mouse
- Real-time recognition of digits from 0-9
- Display of prediction results and confidence level

## System Requirements

- Python 3.9-3.12 (Latest TensorFlow requires Python 3.9-3.12)
- Operating System: Windows, macOS, or Linux

## Installation

### Method 1: Using a Virtual Environment (Recommended)

#### Windows:
1. Run the virtual environment setup script:
```
setup_env.bat
```

2. Activate the virtual environment:
```
venv\Scripts\activate
```

3. Run the application:
```
python main.py
```

#### Linux/macOS:
1. Give execution permission to the script:
```
chmod +x setup_env.sh
```

2. Run the virtual environment setup script:
```
./setup_env.sh
```

3. Activate the virtual environment:
```
source venv/bin/activate
```

4. Run the application:
```
python main.py
```

### Method 2: Direct Installation

1. Install setuptools and wheel (important):
```
pip install setuptools wheel
```

2. Install the required libraries:
```
pip install -r requirements.txt
```

3. Run the application:
```
python main.py
```

## How to Use

1. Draw a digit from 0-9 in the drawing area using the mouse
2. Click the "Predict" button to recognize the digit
3. Click the "Clear" button to clear the drawing area

## Project Structure

- `main.py`: Main file containing the user interface, application logic, and environment checks
- `model.py`: Definition and training of the deep learning model
- `utils.py`: Utility functions for image processing
- `setup.py`: Project installation configuration
- `setup_env.bat`: Virtual environment setup script for Windows
- `setup_env.sh`: Virtual environment setup script for Linux/macOS
- `requirements.txt`: List of required libraries

## Technologies Used

- TensorFlow/Keras: Building and training the model
- Pygame: Creating the drawing interface
- OpenCV and NumPy: Image processing
- MNIST: Training dataset

## Troubleshooting

### Error "Cannot import 'setuptools.build_meta'"
- Install setuptools and wheel first:
```
pip install setuptools wheel
```
- Then install other libraries:
```
pip install -r requirements.txt
```

### Error when installing NumPy or other libraries
- Make sure you have installed setuptools and wheel
- Try installing a specific version of NumPy:
```
pip install numpy==1.24.3
```
- If you still encounter errors, try installing each library separately:
```
pip install tensorflow
pip install numpy
pip install pygame
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

### TensorFlow cannot be installed
- Make sure you are using Python 3.9-3.12
- On Windows, you may need to install Visual C++ Redistributable
- On macOS, you may need to install Xcode Command Line Tools

### Error when running the application
- Make sure all libraries are installed with the correct versions
- Check if the virtual environment is activated
- Run `python main.py` to automatically check for required libraries 