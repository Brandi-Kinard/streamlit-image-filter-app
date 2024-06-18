import streamlit as st
import io
import base64
import cv2
from PIL import Image
from imageFilters import *


# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Set title.
st.title('Artistic Image Filters')

# Upload image.
uploaded_file = st.file_uploader('Choose an image file:', type=['png', 'jpg'])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.header('Original')
        # Display uploaded image.
        st.image(img, channels='BGR', use_column_width=True)

    st.header('Filter Examples:')
    # Display a selection box for choosing the filter to apply.
    option = st.selectbox('Select a filter:',
                          ('None',
                           'Black and White',
                           'Sepia / Vintage',
                           'Vignette Effect',
                           'Pencil Sketch',
                           'Cartoon Style',
                           'Pop Art Dots',
                           'Dithering 1',
                           'Dithering 2',
                           'Dithering 3'
                           ))

    # Define columns for thumbnail images.
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    with col1:
        st.caption('Black and White')
        st.image('filter_bw.png')
    with col2:
        st.caption('Sepia / Vintage')
        st.image('filter_sepia.png')
    with col3:
        st.caption('Vignette Effect')
        st.image('filter_vignette.png')
    with col4:
        st.caption('Pencil Sketch')
        st.image('filter_pencil_sketch.png')
    with col5:
        st.caption('Cartoon Style')
        st.image('filter_cartoon.png')
    with col6:
        st.caption('Pop Art Dots')
        st.image('filter_pop_art_dots.png')
    # Floyd-Steinberg Dithering
    with col7:
        st.caption('Dithering 1')
        st.image('filter_dithering_3.png')
    # Halftone Dithering
    with col8:
        st.caption('Dithering 2')
        st.image('filter_dithering_2.png')
    # Bayer Dithering
    with col9:
        st.caption('Dithering 3')
        st.image('filter_dithering_3.png')


    # Flag for showing output image.
    output_flag = 1
    # Colorspace of output image.
    color = 'BGR'

    # Generate filtered image based on the selected option.
    if option == 'None':
        # Don't show output image.
        output_flag = 0
    elif option == 'Black and White':
        output = bw_filter(img)
        color = 'GRAY'
    elif option == 'Sepia / Vintage':
        output = sepia(img)
    elif option == 'Vignette Effect':
        level = st.slider('level', 0, 5, 2)
        output = vignette(img, level)
    elif option == 'Pencil Sketch':
        ksize = st.slider('Blur kernel size', 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = 'GRAY'
    elif option == 'Cartoon Style':
        output = cartoon_filter(img)
    elif option == 'Pop Art Dots':
        output = polka_dot_art_filter(img)
    elif option == 'Dithering 1':
        output = floyd_steinberg_dithering(img)
        color = 'GRAY'
    elif option == 'Dithering 2':
        output = halftone_dithering(img)
        color = 'GRAY'
    elif option == 'Dithering 3':
        output = bayer_dithering(img)
        color = 'GRAY'

    with output_col:
        if output_flag == 1:
            st.header('Output')
            st.image(output, channels=color)
            # from array convert cv2 image into PIL format for saving it using download link.
            if color == 'BGR':
                result = Image.fromarray(output[:, :, ::-1])
            else:
                result = Image.fromarray(output)
            # Display link.
            st.markdown(get_image_download_link(result, 'output.png', 'Download ' + 'Output'),
                        unsafe_allow_html=True)








