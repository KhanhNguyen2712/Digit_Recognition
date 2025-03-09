#!/usr/bin/env python
"""
Handwritten Digit Recognition Application
This script combines environment checking and the main application.
"""

import importlib.util
import os
import platform
import subprocess
import sys

import cv2
import numpy as np


# Check if pygame is available before importing
def check_module(module_name):
    """Check if a module is installed"""
    return importlib.util.find_spec(module_name) is not None


# Environment checking functions
def check_python_version():
    """Check Python version"""
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor < 9 or minor > 12:
        print(f"WARNING: Current Python version is {major}.{minor}")
        print("Latest TensorFlow requires Python 3.9-3.12")
        print("The application may not work correctly with the current Python version.")
        return False
    return True


def check_environment():
    """Check environment and required libraries"""
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("\nContinuing to run the application, but errors may occur...\n")

    # Check required libraries
    required_modules = ["tensorflow", "numpy", "pygame", "cv2", "sklearn", "matplotlib"]
    missing_modules = []

    for module in required_modules:
        if not check_module(module):
            missing_modules.append(module)

    if missing_modules:
        print("Missing libraries:", ", ".join(missing_modules))
        print("\nPlease install the missing libraries:")

        # Check if in virtual environment
        in_venv = sys.prefix != sys.base_prefix

        if in_venv:
            print(
                "You are in a virtual environment. Run the following command to install:"
            )
            print("pip install -r requirements.txt")

            # Suggest installing setuptools if not present
            if not check_module("setuptools"):
                print("\nREQUIRED: Install setuptools first:")
                print("pip install setuptools wheel")
        else:
            print(
                "You are not in a virtual environment. We recommend using a virtual environment:"
            )
            if os.name == "nt":  # Windows
                print("1. Run: setup_env.bat")
                print("2. Activate environment: venv\\Scripts\\activate")
            else:  # Linux/macOS
                print("1. Run: ./setup_env.sh")
                print("2. Activate environment: source venv/bin/activate")

        return False

    # Check TensorFlow version
    if check_module("tensorflow"):
        import tensorflow as tf

        tf_version = tf.__version__
        print(f"TensorFlow version: {tf_version}")
        major, minor = map(int, tf_version.split(".")[:2])
        if major < 2 or (major == 2 and minor < 16):
            print(f"WARNING: This project is designed for TensorFlow 2.16 or later")
            print(f"Current version is {tf_version}, compatibility issues may occur.")

    # Check model file
    model_path = "improved_digit_model.keras"
    legacy_model_path = "digit_model.keras"

    if os.path.exists(model_path):
        print(f"Model exists at {model_path}")
    elif os.path.exists(legacy_model_path):
        print(
            f"Found legacy model at {legacy_model_path}. Will convert to new format when running."
        )
    else:
        print(
            f"Model does not exist. A new model will be trained when running the application."
        )

    return True


# Import application modules if environment check passes
if check_module("pygame") and check_module("tensorflow"):
    import pygame
    from tensorflow.keras.models import load_model

    from model import load_trained_model, predict_digit
    from utils import draw_text, prepare_for_prediction

    # Application constants
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    DRAWING_AREA_SIZE = 280
    DRAWING_AREA_POSITION = (50, 150)
    LINE_THICKNESS = 15

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_GRAY = (230, 230, 230)
    BLUE = (0, 120, 255)
    RED = (255, 0, 0)
    GREEN = (0, 180, 0)

    # Application variables
    screen = None
    drawing_surface = None
    model = None

    def initialize_app():
        """Initialize pygame and application surfaces"""
        global screen, drawing_surface

        # Initialize pygame
        pygame.init()

        # Create window
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Handwritten Digit Recognition")

        # Create drawing surface
        drawing_surface = pygame.Surface((DRAWING_AREA_SIZE, DRAWING_AREA_SIZE))
        drawing_surface.fill(WHITE)

    def load_model():
        """Load model from file"""
        global model
        model = load_trained_model()
        return model is not None

    def clear_drawing():
        """Clear the drawing area"""
        drawing_surface.fill(WHITE)

    def get_drawing_as_array():
        """Convert drawing surface to numpy array"""
        # Convert surface to numpy array
        drawing_array = pygame.surfarray.array3d(drawing_surface)

        # Convert from RGB to BGR (for OpenCV)
        drawing_array = drawing_array.transpose([1, 0, 2])

        # Convert to grayscale
        return cv2.cvtColor(drawing_array, cv2.COLOR_RGB2GRAY)

    def predict():
        """Predict digit from drawing"""
        if model is None:
            return None, 0

        # Get image from drawing area
        drawing_array = get_drawing_as_array()

        # Prepare image for prediction
        processed_image = prepare_for_prediction(drawing_array)

        # Predict
        return predict_digit(model, processed_image)

    def draw_button(rect, text, color=LIGHT_GRAY, hover_color=GRAY, text_color=BLACK):
        """Draw button with hover effect"""
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        button_color = hover_color if is_hovered else color

        # Draw button
        pygame.draw.rect(screen, button_color, rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, rect, 2, border_radius=5)

        # Draw text
        font = pygame.font.Font(None, 30)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)

        return is_hovered

    def handle_drawing(event, drawing, last_pos):
        """Handle drawing logic"""
        drawing_rect = pygame.Rect(
            DRAWING_AREA_POSITION, (DRAWING_AREA_SIZE, DRAWING_AREA_SIZE)
        )

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if drawing_rect.collidepoint(event.pos):
                drawing = True
                last_pos = (
                    event.pos[0] - DRAWING_AREA_POSITION[0],
                    event.pos[1] - DRAWING_AREA_POSITION[1],
                )

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False
            last_pos = None

        elif event.type == pygame.MOUSEMOTION and drawing and last_pos is not None:
            # Calculate relative position in drawing area
            current_pos = (
                event.pos[0] - DRAWING_AREA_POSITION[0],
                event.pos[1] - DRAWING_AREA_POSITION[1],
            )

            # Check if position is within drawing area
            if (
                0 <= current_pos[0] < DRAWING_AREA_SIZE
                and 0 <= current_pos[1] < DRAWING_AREA_SIZE
            ):
                # Draw line
                pygame.draw.line(
                    drawing_surface, BLACK, last_pos, current_pos, LINE_THICKNESS
                )
                last_pos = current_pos

        return drawing, last_pos

    def render_ui(prediction, probability):
        """Render the user interface"""
        # Clear screen
        screen.fill(WHITE)

        # Draw title
        draw_text(screen, "Handwritten Digit Recognition", (250, 50), font_size=40)
        draw_text(screen, "Draw a digit (0-9):", (50, 120), font_size=30)

        # Draw drawing area frame
        pygame.draw.rect(
            screen,
            BLACK,
            (
                DRAWING_AREA_POSITION[0] - 2,
                DRAWING_AREA_POSITION[1] - 2,
                DRAWING_AREA_SIZE + 4,
                DRAWING_AREA_SIZE + 4,
            ),
            2,
        )

        # Display drawing area
        screen.blit(drawing_surface, DRAWING_AREA_POSITION)

        # Draw buttons
        predict_button = pygame.Rect(350, 250, 150, 50)
        clear_button = pygame.Rect(350, 320, 150, 50)

        predict_hovered = draw_button(
            predict_button,
            "Predict",
            color=(0, 100, 220),
            hover_color=BLUE,
            text_color=WHITE,
        )
        clear_hovered = draw_button(clear_button, "Clear")

        # Display prediction results
        if prediction is not None:
            result_text = f"Prediction: {prediction}"
            confidence_text = f"Confidence: {probability:.2f}"

            draw_text(screen, result_text, (350, 400), font_size=36, color=(0, 100, 0))
            draw_text(screen, confidence_text, (350, 450), font_size=30)

            # Display processed image
            processed_img = get_drawing_as_array()
            if processed_img is not None:
                # Resize for display
                display_img = cv2.resize(processed_img, (140, 140))

                # Convert to pygame surface
                surf = pygame.surfarray.make_surface(display_img)
                screen.blit(surf, (550, 150))

                draw_text(screen, "Processed Image:", (550, 120), font_size=24)

        # Update screen
        pygame.display.flip()

        return predict_button, clear_button

    def run_application():
        """Run the main application"""
        # Initialize pygame and surfaces
        initialize_app()

        # Load model
        print("Loading model...")
        model_loaded = load_model()
        if not model_loaded:
            print("Could not load model!")
            return

        # State variables
        running = True
        drawing = False
        last_pos = None
        prediction = None
        probability = 0

        # Main loop
        while running:
            # Render UI
            predict_button, clear_button = render_ui(prediction, probability)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue

                # Handle mouse events for drawing
                drawing, last_pos = handle_drawing(event, drawing, last_pos)

                # Handle button clicks
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check Predict button
                    if predict_button.collidepoint(event.pos):
                        prediction, probability = predict()

                    # Check Clear button
                    elif clear_button.collidepoint(event.pos):
                        clear_drawing()
                        prediction = None
                        probability = 0

        # End pygame
        pygame.quit()


def main():
    """Main entry point for the application"""
    print("Checking environment...")
    if check_environment():
        print("Running handwritten digit recognition application...")
        try:
            run_application()
        except Exception as e:
            print(f"Error running application: {e}")
            print("\nPlease check your installation and try again.")
    else:
        print("Environment check failed. Please install the required dependencies.")


if __name__ == "__main__":
    main()
