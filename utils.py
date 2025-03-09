import cv2
import numpy as np


def preprocess_image(image, target_size=(28, 28)):
    """
    Preprocess image to match model input requirements

    Args:
        image: Input image (numpy array)
        target_size: Desired output size

    Returns:
        Processed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert colors (white on black -> black on white)
    image = 255 - image

    # Find digit contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour (digit)
        max_contour = max(contours, key=cv2.contourArea)

        # Create bounding box
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop image to bounding box
        digit = image[y : y + h, x : x + w]

        # Add padding to ensure digit is centered
        aspect_ratio = h / w

        if aspect_ratio > 1:  # Taller than wide
            # Add padding to left and right
            new_w = h
            padding = int((new_w - w) / 2)
            digit_square = np.zeros((h, new_w), dtype=np.uint8)
            digit_square[:, padding : padding + w] = digit
        else:  # Wider than tall
            # Add padding to top and bottom
            new_h = w
            padding = int((new_h - h) / 2)
            digit_square = np.zeros((new_h, w), dtype=np.uint8)
            digit_square[padding : padding + h, :] = digit

        # Resize to desired dimensions
        processed_digit = cv2.resize(
            digit_square, target_size, interpolation=cv2.INTER_AREA
        )

        # Smooth image
        processed_digit = cv2.GaussianBlur(processed_digit, (3, 3), 0)

        # Normalize pixel values to [0, 1]
        processed_digit = processed_digit.astype("float32") / 255.0

        return processed_digit

    # If no contour found, return empty image
    return np.zeros(target_size, dtype="float32")


def prepare_for_prediction(image):
    """
    Prepare image for prediction

    Args:
        image: Input image

    Returns:
        Processed and formatted image for model input
    """
    processed = preprocess_image(image)
    return np.expand_dims(np.expand_dims(processed, axis=-1), axis=0)


def draw_text(surface, text, position, font_size=24, color=(0, 0, 0)):
    """
    Draw text on pygame surface

    Args:
        surface: Pygame surface to draw on
        text: Text to draw
        position: Position (x, y)
        font_size: Font size
        color: Text color
    """
    import pygame

    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)
