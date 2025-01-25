import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
import struct

def sorted_directory_listing(directory):
    """
    Returns a sorted list of filenames from the specified directory.

    Parameters:
    directory (str): The path to the directory to list files from.

    Returns:
    list: A sorted list of filenames present in the directory.
    """
    items = os.listdir(directory)
    sorted_items = sorted(items)
    return sorted_items




def image_reading(filepath):
    """
    Reads an image from the given file path in grayscale mode.

    Parameters:
    filepath (str): The path to the image file.

    Returns:
    numpy.ndarray: The loaded image as a NumPy array in grayscale.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return image



def apply_convolution(image, filter: str):
    """
    Applies a convolution filter to an image using a predefined kernel.

    Parameters:
    image (numpy.ndarray): The input image to be processed.
    filter (str): The type of filter to apply. Options are:
        - 'box': Enhances edges using a box-like kernel.
        - 'numbers': Enhances features using a different edge-detection kernel.

    Returns:
    numpy.ndarray: The processed image with the applied convolution and inversion.
    """
    if filter == 'box':
        kernel = np.array([[0, 1, 2],
                           [1, -4, 1],
                           [2, 1, 0]])
        
        edges_x = convolved_image = cv2.filter2D(image, -1, kernel)
        inverted_image = 255 - edges_x
        return inverted_image

    elif filter == 'numbers':
        kernel = np.array([[-0.5, 0, 1],
                           [-2, 0, 2],
                           [1, 0, 1]])
                
        edges_x = cv2.filter2D(image, -1, kernel)
        inverted_image = 255 - edges_x
        return inverted_image


def slice_rotate_image(image, height: float, length: float, delta_height: float, delta_length: float, digit_width: float):
    """
    Processes an image by applying convolution, detecting contours, rotating to correct orientation, 
    and slicing the image into six digit segments.

    Parameters:
    image (numpy.ndarray): Input grayscale image containing digits.
    height (float): Expected height of the target object in the image.
    length (float): Expected length of the target object in the image.
    delta_height (float): Vertical offset to define digit slices.
    delta_length (float): Horizontal offset to define digit slices.
    digit_width (float): Width of each digit in the image.

    Returns:
    list of numpy.ndarray: A list containing six extracted and resized digit images.
    """

    pre_rotated_image = apply_convolution(image=image, filter='box')  # Image processed to detect rotation

    def find_coordinates(image, reduce: bool = False):
        """
        Finds the bounding box of the largest contour in the image.

        Parameters:
        image (numpy.ndarray): Input preprocessed image.
        reduce (bool): If True, returns only width and height of the bounding box.
                       If False, returns x, y, width, and height.

        Returns:
        tuple: (w, h) if reduce is True, otherwise (x, y, w, h).
        """
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        if reduce:
            _, _, w, h = cv2.boundingRect(largest_contour)
            return w, h
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            return x, y, w, h

    def rotate(angle, image):
        """
        Rotates an image by a given angle.

        Parameters:
        angle (float): The angle in degrees to rotate the image.
        image (numpy.ndarray): The input image to be rotated.

        Returns:
        numpy.ndarray: The rotated image.
        """
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated_image

    w, h = find_coordinates(image=pre_rotated_image, reduce=True)

    # Calculate initial rotation angle
    angle = np.arctan(h / w) - np.arctan(height / length)
    angle = np.rad2deg(angle)
    rotated_image = rotate(angle=angle, image=image)

    # Check if rotation was successful
    rotated_image_check = apply_convolution(image=rotated_image, filter='box')
    w2, h2 = find_coordinates(image=rotated_image_check, reduce=True)

    # Validate rotation and correct if necessary
    angle2 = np.arctan(h2 / w2) - np.arctan(height / length)
    angle2 = np.rad2deg(angle2)
    if round(angle2) >= 4.5:
        angle = -angle
        rotated_image = rotate(angle=angle, image=image)

    rotated_image = apply_convolution(rotated_image, filter='box')

    # Extract region of interest
    x, y, w, h = find_coordinates(image=rotated_image, reduce=False)
    reduced_image = rotated_image[y:y+h, x:x+w]

    # Compute ratio for resizing
    ratio = reduced_image.shape[0] / height

    # Define slices for individual digits
    digit_slice = {}
    w_d = 0
    for i in range(6):
        h_d = delta_height * ratio
        w_d += ratio * delta_length
        digit = digit_width * ratio
        digit_slice[i+1] = (slice(int(h_d), int(reduced_image.shape[0]-h_d)), slice(int(w_d + i * digit), int(w_d + i * digit + digit)))
        w_d += delta_length * ratio

    # Apply final rotation and extract digits
    reduced_image = rotate(angle, image=apply_convolution(image, filter='numbers'))
    reduced_image = reduced_image[y:y+h, x:x+w]

    digit_1 = cv2.resize(reduced_image[digit_slice[1]], (30, 50))
    digit_2 = cv2.resize(reduced_image[digit_slice[2]], (30, 50))
    digit_3 = cv2.resize(reduced_image[digit_slice[3]], (30, 50))
    digit_4 = cv2.resize(reduced_image[digit_slice[4]], (30, 50))
    digit_5 = cv2.resize(reduced_image[digit_slice[5]], (30, 50))
    digit_6 = cv2.resize(reduced_image[digit_slice[6]], (30, 50))

    return [digit_1, digit_2, digit_3, digit_4, digit_5, digit_6]


def hexdecimal_to_picture_array(HEXADECIMAL_BYTES):
    """
    Converts a list of hexadecimal byte values representing pixel data into a QCIF resolution (144x176) RGB image array.

    Parameters:
    HEXADECIMAL_BYTES (list of bytes): A list containing 16-bit hexadecimal values representing pixel data in RGB565 format.

    Returns:
    numpy.ndarray: A 3D NumPy array of shape (144, 176, 3) representing the image in RGB format.
    """

    raw_bytes = np.array(HEXADECIMAL_BYTES, dtype="uint16")
    image = np.zeros((len(raw_bytes), 3), dtype=int)

    # Loop through all pixels and form the image
    for i in range(len(raw_bytes)):
        # Read 16-bit pixel
        pixel = struct.unpack('>h', raw_bytes[i])[0]

        # Convert RGB565 to RGB 24-bit
        r = ((pixel >> 11) & 0x1F) << 3
        g = ((pixel >> 5) & 0x3F) << 2
        b = ((pixel >> 0) & 0x1F) << 3
        image[i] = [r, g, b]

    # Reshape the image to QCIF resolution (144x176)
    image = np.reshape(image, (144, 176, 3))

    return image