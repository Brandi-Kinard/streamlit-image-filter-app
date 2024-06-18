import cv2
import numpy as np
import streamlit as st

# new
@st.cache_data
def cartoon_filter(img, sigma_s=50):
    dst = cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=0.25)
    edges = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    colored = cv2.bilateralFilter(dst, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.bitwise_and(colored, colored, mask=edges)


# new
@st.cache_data
def polka_dot_art_filter(img):
    # Initialize a white background
    black_background = np.ones_like(img) * 0

    # Define the size of each dot
    dot_size = 10  # Size of each dot

    # Applying the dot effect
    for y in range(0, img.shape[0], dot_size):
        for x in range(0, img.shape[1], dot_size):
            block = img[y:y + dot_size, x:x + dot_size]
            if block.size > 0:
                avg_color = block.mean(axis=0).mean(axis=0)
                cv2.circle(black_background, (x + dot_size // 2, y + dot_size // 2), dot_size // 2, avg_color, -1)

    return black_background



# new
@st.cache_data
def floyd_steinberg_dithering(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y in range(gray.shape[0]-1):
        for x in range(1, gray.shape[1]-1):
            old_pixel = gray[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            gray[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            gray[y, x+1] += quant_error * 5 / 16
            gray[y+1, x-1] += quant_error * 3 / 16
            gray[y+1, x] += quant_error * 7 / 16
            gray[y+1, x+1] += quant_error * 1 / 16
    return gray



# new
'''
@st.cache
def halftone_dithering(img):
    # Custom function for halftone dithering
    # Placeholder function here, you'll need to implement based on the reference or your own approach
    pass
'''
@st.cache_data
def halftone_dithering(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a dot pattern effect
    dither = np.zeros_like(gray)
    for y in range(0, gray.shape[0], 4):
        for x in range(0, gray.shape[1], 4):
            block = gray[y:y + 4, x:x + 4]
            avg = np.mean(block)
            radius = int(avg / 255.0 * 4)  # Darker areas have bigger dots
            cv2.circle(dither, (x + 2, y + 2), radius, (255, 255, 255), -1)
    return dither


# new
'''
@st.cache
def bayer_dithering(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Bayer dithering
    # Placeholder function here, you'll need to implement based on the reference or your own approach
    pass
'''
@st.cache_data
def bayer_dithering(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bayer matrix 4x4
    bayer_matrix = np.array([[0, 8, 2, 10],
                             [12, 4, 14, 6],
                             [3, 11, 1, 9],
                             [15, 7, 13, 5]]) * 16

    # Tile the Bayer matrix
    bayer_pattern = np.tile(bayer_matrix, (int(gray.shape[0] / 4) + 1, int(gray.shape[1] / 4) + 1))
    bayer_pattern = bayer_pattern[:gray.shape[0], :gray.shape[1]]

    # Scale the grayscale image
    scaled_gray = (gray / 256) * 255

    # Apply Bayer dithering
    dithered = np.where(scaled_gray < bayer_pattern, 0, 255).astype(np.uint8)

    return dithered


'''
@st.cache_data
def ascii_art_filter(img, scale=0.1, cols=80):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust the number of columns based on image width and scale
    width = gray.shape[1]
    cell_width = width / cols
    cell_height = cell_width / scale  # Maintain aspect ratio
    rows = int(gray.shape[0] / cell_height)

    # ASCII characters set
    ascii_chars = "@%#*+=-:. "

    # Prepare to hold the ASCII art
    ascii_art = []

    # Generate ASCII art
    for i in range(rows):
        line = []
        for j in range(cols):
            # Ensure the coordinate does not exceed image dimensions
            x, y = int(j * cell_width), int(i * cell_height)
            if x < width and y < gray.shape[0]:
                pixel_value = gray[y, x]
                # Convert pixel value to an index in the ascii_chars string
                index = int(pixel_value / 255 * (len(ascii_chars) - 1))
                line.append(ascii_chars[index])
        ascii_art.append("".join(line))

    return "\n".join(ascii_art)
'''


@st.cache_data
def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

@st.cache_data
def vignette(img, level=2):
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Apply the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette

@st.cache_data
def sepia(img):
        img_sepia = img.copy()
        # Converting to RGB as sepia matrix below is for RGB.
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
        img_sepia = np.array(img_sepia, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                        [0.349, 0.686, 0.168],
                                                        [0.272, 0.534, 0.131]]))
        # Clip values to the range [0, 255].
        img_sepia = np.clip(img_sepia, 0, 255)
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
        return img_sepia


@st.cache_data
def pencil_sketch(img, ksize=5):
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch



