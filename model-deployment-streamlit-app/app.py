import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

# specify the path of the model
model_path = 'C:/Users/100790606/notebook_files/cartridge-segmentation/saved-model/unet_model.h5'
model = load_model(model_path)

tf.get_logger().setLevel('ERROR') # suppress tensorflow warnings

def main():
    st.title("Cartridge Image Segmentation")
    st.header("Upload a cartridge case image to perform masking.")
    st.subheader("Please select an image below")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns(2)  # Create two columns

        with col1:
            st.image(uploaded_file, use_column_width=True, width=256)
            st.markdown(get_html_caption("Uploaded Image"), unsafe_allow_html=True)

        with col2:
            processed_image = process_image(uploaded_file)
            st.image(processed_image, use_column_width=True, width=256)
            st.markdown(get_html_caption("Generated Image"), unsafe_allow_html=True)

def get_html_caption(text):
    return f"<h3 style='text-align:center; color: black;'>{text}</h3>"

# function to perform post-processing after generating image from the model
def post_process(new_input_img, output_img2):
    img3 = cv2.subtract(new_input_img*255,output_img2*255)
    img_op = cv2.dilate(img3, None,iterations = 3)
    img_op = cv2.erode(img_op, None,iterations = 3)

    img_op = cv2.dilate(img_op, None,iterations = 3)
    img_op = cv2.erode(img_op, None,iterations = 3)

    img_op_clip = np.clip(img_op, 0.0, 1.0)
    img_op_clip = 1 - img_op_clip
    img_op_clip[img_op_clip > 0] = 1

    img_op_clip_bin = cv2.cvtColor(img_op_clip, cv2.COLOR_BGR2GRAY)
    img_op_clip_bin[img_op_clip_bin > 0] = 1

    aa = new_input_img.copy() 
    aa[:,:,0] = aa[:,:,0]*img_op_clip_bin
    aa[:,:,1] = aa[:,:,1]*img_op_clip_bin
    aa[:,:,2] = aa[:,:,2]*img_op_clip_bin
    
    kernel = np.ones((15,15), np.uint8)
    closing_result = cv2.morphologyEx(aa, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((30,30), np.uint8)
    closing_result = cv2.morphologyEx(closing_result, cv2.MORPH_CLOSE, kernel)
    closing_result[closing_result > 0] = 1
    result_image = new_input_img.copy()
    result_image[closing_result == 1] = output_img2[closing_result == 1]
    result_image[closing_result == 0] = new_input_img[closing_result == 0]
    return result_image

# function to process the uploaded image using the loaded U-Net model
def process_image(uploaded_file):
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    img_array = cv2.resize(img_array, (256, 256))
    
    img_array = (img_array / 255.0).astype(np.float32)  
    print(img_array.min(), img_array.max())
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    result = post_process(img_array, predictions[0])
    return result

if __name__ == "__main__":
    main()
