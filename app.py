import streamlit as st
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import pandas as pd

st.header("Tooth detection and segmentation in panoramic X-rays")
st.subheader("Demo improvement iteration")
st.markdown("Demo for testing an image segmentation CNN model")

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

# Load the pretrained U-Net model.
model_id = "SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net"
model = from_pretrained_keras(model_id)

image_file = st.file_uploader("Upload your image here.", type=["png", "jpg", "jpeg"])


def to_grayscale(image):
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


# Example files for quick local testing.
example_images = ["teeth_1.png", "teeth_2.png", "teeth_3.png"]

col1, col2, col3 = st.columns(3)

with col1:
    example = Image.open(example_images[0])
    st.image(example, width=200)
    if st.button("Run this example 1"):
        image_file = example_images[0]

with col2:
    example = Image.open(example_images[1])
    st.image(example, width=200)
    if st.button("Run this example 2"):
        image_file = example_images[1]

with col3:
    example = Image.open(example_images[2])
    st.image(example, width=200)
    if st.button("Run this example 3"):
        image_file = example_images[2]

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, width=850)

    with st.spinner("Analyzing panoramic X-ray. This may take a few seconds..."):
        image = np.array(image)

        # The model expects one grayscale channel at 512 x 512.
        model_input = to_grayscale(image)
        model_input = cv2.resize(
            model_input, (512, 512), interpolation=cv2.INTER_LANCZOS4
        )
        model_input = np.float32(model_input / 255)
        model_input = np.reshape(model_input, (1, 512, 512, 1))

        prediction = model.predict(model_input)[0]

        prediction = cv2.resize(
            prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )

        mask = np.uint8(prediction * 255)
        _, mask = cv2.threshold(
            mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((5, 5), dtype=np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmented_image = cv2.drawContours(to_rgb(image), contours, -1, (255, 0, 0), 3)

    if segmented_image is not None:
        st.subheader("Image segmentation:")
        st.image(segmented_image, width=850)

        st.subheader("Diagnostic metrics overview")

        tooth_count = len(contours)
        total_area = np.sum(mask > 0)
        average_area = total_area / tooth_count if tooth_count > 0 else 0

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Estimated tooth count", tooth_count)
        col_m2.metric("Total dental area (px)", f"{total_area:,}")
        col_m3.metric("Average tooth area (px)", f"{int(average_area):,}")

        st.markdown("### Detected instances data")

        tooth_data = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            _, _, w, h = cv2.boundingRect(contour)
            tooth_data.append({
                "ID": i + 1,
                "Area (px)": area,
                "Width (px)": w,
                "Height (px)": h
            })

        teeth_df = pd.DataFrame(tooth_data)
        st.dataframe(teeth_df, use_container_width=True)
