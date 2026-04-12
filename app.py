import streamlit as st
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import pandas as pd

st.header("Tooth detection and segmentation in panoramic X-Rays")
st.subheader("Iteration to improve demo")
st.markdown(
    """
    Demo for testing image segmentation CNN model
"""
)

st.markdown(
    """
    ### Technical overview
    
    * **Architecture:** It utilizes the U-Net architecture, a popular "encoder-decoder" convolutional neural network (CNN) specifically optimized for biomedical image segmentation where pixel-level accuracy is critical.
    * **Performance:** In the accompanying research, the model achieved a Dice overlap score of 95.4% for overall teeth segmentation.
    * **Post-processing:** A key highlight of this specific implementation is the use of grayscale morphological filtering and operations applied to the sigmoid output. This reduces tooth counting errors significantly (from 26.8% down to roughly 6.2%).
    * **Dataset:** The model was trained on a relatively small but highly curated dataset (approximately 105 to 116 panoramic images) based on work by Abdi et al. (2015).
    
    ### Key applications
    
    * **Clinical diagnosis:** Assists dentists in identifying the boundaries of individual teeth to detect caries, lesions, or bone loss.
    * **Forensics and identification:** Automates the process of identifying dental patterns for human remains or age/gender determination.
    * **Treatment planning:** Provides a baseline for orthodontic therapy workups by isolating dental structures from the surrounding mandible and maxilla.
    """
)

## Select and load the model
model_id = "SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net"
model = from_pretrained_keras(model_id)

## Allow the user to upload an image
archivo_imagen = st.file_uploader("Upload your image here.", type=["png", "jpg", "jpeg"])

## If an image has more than one channel, it is converted to grayscale (1 channel)
def convertir_one_channel(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        return img

def convertir_rgb(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        return img

## We will manipulate the interface so we can use example images
## If the user clicks on an example, the model will run with the following:
ejemplos = ["dientes_1.png", "dientes_2.png", "dientes_3.png"]

## Create three columns; an example image will be in each one
col1, col2, col3 = st.columns(3)

with col1:
    ## The image is loaded and displayed in the interface
    ex = Image.open(ejemplos[0])
    st.image(ex, width=200)
    ## If the button is pressed, we will use this example in the model
    if st.button("Run this example 1"):
        archivo_imagen = ejemplos[0]

with col2:
    ex1 = Image.open(ejemplos[1])
    st.image(ex1, width=200)
    if st.button("Run this example 2"):
        archivo_imagen = ejemplos[1]

with col3:
    ex2 = Image.open(ejemplos[2])
    st.image(ex2, width=200)
    if st.button("Run this example 3"):
        archivo_imagen = ejemplos[2]

## If we have an image to input into the model, 
## we process it and feed it to the model
if archivo_imagen is not None:
    ## Load the image with PIL and display it immediately
    img = Image.open(archivo_imagen)
    st.image(img, width=850)
    
    ## Trigger the loading spinner for the heavy lifting
    with st.spinner('Analyzing panoramic X-ray. This may take a few seconds...'):
        img = np.array(img)    #Creates a writable copy

        ## Process the image for model input
        img_cv = convertir_one_channel(img)
        img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        img_cv = np.float32(img_cv / 255)
        img_cv = np.reshape(img_cv, (1, 512, 512, 1))

        ## Feed the NumPy array into the model
        predicted = model.predict(img_cv)
        predicted = predicted[0]

        ## Resize the image back to its original shape and add the segmentation masks
        predicted = cv2.resize(
            predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        
        mask = np.uint8(predicted * 255)
        _, mask = cv2.threshold(
            mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        kernel = np.ones((5, 5), dtype=np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        cnts, hieararch = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.drawContours(convertir_one_channel(img), cnts, -1, (255, 0, 0), 3)

    ## If we successfully obtained a result, display it in the interface
    if output is not None:
        st.subheader("Image segmentation:")
        st.write(output.shape)
        st.image(output, width=850)

        st.subheader("Diagnostic metrics overview")
        
        conteo_dientes = len(cnts)
        area_total = np.sum(mask > 0)
        area_promedio = area_total / conteo_dientes if conteo_dientes > 0 else 0
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Estimated tooth count", conteo_dientes)
        col_m2.metric("Total dental area (px)", f"{area_total:,}")
        col_m3.metric("Average tooth area (px)", f"{int(area_promedio):,}")
        
        st.markdown("### Detected instances data")
        
        datos_dientes = []
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            datos_dientes.append({
                "ID": i + 1,
                "Area (px)": area,
                "Width (px)": w,
                "Height (px)": h
            })
            
        df_dientes = pd.DataFrame(datos_dientes)
        st.dataframe(df_dientes, use_container_width=True)
        