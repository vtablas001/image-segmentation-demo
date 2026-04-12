# Tooth Segmentation in Panoramic X-Rays

I created this interactive Streamlit application that performs automatic semantic segmentation of teeth in panoramic dental X-ray images using a pretrained U-Net Convolutional Neural Network. You can upload a panoramic radiograph or use one of the built-in examples to instantly visualize segmented tooth boundaries, estimated tooth counts, and per-tooth diagnostic metrics.

[![Live Demo on HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/vtablas001/image_segmentation)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [What Is Image Segmentation?](#what-is-image-segmentation)
- [About the Model](#about-the-model)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Key Applications](#key-applications)
- [Technologies](#technologies)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [Author](#author)

---

## Overview

Panoramic dental X-rays (orthopantomograms) are one of the most commonly used imaging modalities in dentistry. Manually identifying and delineating individual teeth within these images is time-consuming and subject to inter-observer variability. This project demonstrates how deep learning can automate that process, delivering pixel-level tooth segmentation with a single inference pass.

The application loads a pretrained **U-Net** model from Hugging Face Hub, processes the input image through a complete computer vision pipeline, and displays the segmented output alongside quantitative diagnostic metrics.

---

## What Is Image Segmentation?

**Image segmentation** is a computer vision task that assigns a class label to every pixel in an image, effectively partitioning the image into meaningful regions. Unlike object detection (which draws bounding boxes) or classification (which labels the entire image), segmentation produces a dense, pixel-level map that precisely outlines the boundaries of each structure.

In the context of medical imaging, **semantic segmentation** is particularly valuable because clinical decisions often depend on the exact shape, area, and spatial relationships of anatomical structures.

┌─────────────────────────────────────────────────────────┐
│                    Input X-Ray Image                    │
│                                                         │
│    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│    │Tooth1│  │Tooth2│  │Tooth3│  │Tooth4│  │Tooth5│   │
│    └──────┘  └──────┘  └──────┘  └──────┘  └──────┘   │
│                                                         │
│         U-Net Encoder  >>>  Decoder + Skip Connections  │
│                                                         │
│    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│    │ Mask │  │ Mask │  │ Mask │  │ Mask │  │ Mask │   │
│    └──────┘  └──────┘  └──────┘  └──────┘  └──────┘   │
│                                                         │
│                  Segmented Output + Metrics              │
└─────────────────────────────────────────────────────────┘


---

## About the Model

This demo uses the pretrained model [`SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net`](https://huggingface.co/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net), developed by Selahattin Serdar Helli and Andac Hamamci at the Department of Biomedical Engineering, Yeditepe University, Istanbul, Turkey.

| Detail                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Architecture**      | U-Net (encoder-decoder CNN with skip connections)                           |
| **Framework**         | TensorFlow / Keras                                                          |
| **Input Resolution**  | 512 x 512 pixels, single-channel grayscale                                  |
| **Output**            | Binary segmentation mask (tooth vs. background)                             |
| **Dice Score**        | 95.4% overlap on the evaluation set                                         |
| **Training Dataset**  | 105 to 116 curated panoramic X-ray images (Abdi et al., 2015)              |
| **Post-processing**   | Otsu thresholding + morphological open/close operations                     |
| **Error Reduction**   | Tooth counting error reduced from 26.8% to 6.2% via morphological filtering|

The **U-Net** architecture (Ronneberger et al., 2015) was originally designed for biomedical image segmentation. Its symmetric encoder-decoder structure with skip connections allows the network to capture both high-level semantic context and fine-grained spatial detail, making it especially effective for tasks that require precise boundary delineation.

---

## Features

The application provides a complete end-to-end experience for dental X-ray analysis.

**Image Input** allows users to upload their own panoramic X-ray in PNG, JPG, or JPEG format, or select from three built-in example images for immediate testing.

**Real-time Segmentation** processes the uploaded image through the U-Net model and overlays detected tooth contours directly on the original image.

**Diagnostic Metrics** displays three key measurements at a glance: estimated tooth count, total dental area in pixels, and average tooth area in pixels.

**Detected Instances Table** presents a detailed breakdown of every detected tooth, including its individual area, bounding box width, and bounding box height, displayed in an interactive dataframe.

---

## Pipeline Architecture

The inference pipeline follows five stages:

1. **Preprocessing**: The input image is converted to single-channel grayscale, resized to 512x512, and normalized to the [0, 1] range.

2. **Model Inference**: The preprocessed image is passed through the U-Net, which outputs a probability map where each pixel value represents the likelihood of belonging to a tooth.

3. **Thresholding**: Otsu's adaptive thresholding converts the continuous probability map into a binary mask.

4. **Morphological Refinement**: Opening and closing operations with a 5x5 kernel remove small noise artifacts and fill gaps within tooth regions.

5. **Contour Extraction and Visualization**: OpenCV's `findContours` identifies individual tooth boundaries, which are drawn on the original image and used to compute per-tooth metrics.

```python
# Core inference logic (simplified)
model = from_pretrained_keras(model_id)

img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
img_cv = np.float32(img_cv / 255).reshape(1, 512, 512, 1)

predicted = model.predict(img_cv)[0]
predicted = cv2.resize(predicted, (original_w, original_h))

mask = np.uint8(predicted * 255)
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), dtype=np.float32)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/vtablas001/image-segmentation-demo.git
cd image-segmentation-demo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

### Requirements

```text
streamlit
huggingface_hub==0.21.4
tensorflow==2.15.0
opencv-python-headless
pandas
```

---

## Project Structure

image-segmentation-demo/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── dientes_1.png        # Example panoramic X-ray 1
├── dientes_2.png        # Example panoramic X-ray 2
├── dientes_3.png        # Example panoramic X-ray 3
├── .gitattributes       # Git LFS configuration
└── README.md            # Project documentation

---

## Key Applications

**Clinical Diagnosis**: Assists dental professionals in identifying tooth boundaries to detect caries, periapical lesions, or alveolar bone loss with greater consistency.

**Forensic Identification**: Automates the extraction of dental patterns used in forensic odontology for human identification, age estimation, or gender determination.

**Treatment Planning**: Provides a quantitative baseline for orthodontic and prosthodontic planning by isolating individual dental structures from the mandible and maxilla.

**Education and Research**: Serves as a teaching tool for students and researchers exploring medical image segmentation, U-Net architectures, and computer vision pipelines.

---

## Technologies

- **[TensorFlow / Keras](https://www.tensorflow.org/)**: Deep learning framework for loading and running the pretrained U-Net model.
- **[OpenCV](https://opencv.org/)**: Computer vision library for image preprocessing, thresholding, morphological operations, and contour extraction.
- **[Streamlit](https://streamlit.io/)**: Web application framework for building interactive ML demos.
- **[Hugging Face Hub](https://huggingface.co/)**: Model hosting and deployment platform.
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation library for structuring diagnostic metrics.
- **[NumPy](https://numpy.org/)**: Numerical computing library for array operations and image manipulation.

---

## References

- Abdi, A. H., Kasaei, S., & Mehdizadeh, M. (2015). Automatic segmentation of mandible in panoramic x-ray. *Journal of Medical Imaging, 2*(4), 044003. https://doi.org/10.1117/1.JMI.2.4.044003

- Helli, S. S., & Hamamci, A. (2022). *Segmentation of teeth in panoramic X-ray image using U-Net* [Pretrained model]. Hugging Face. https://huggingface.co/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)* (pp. 234-241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28

---

## Acknowledgments

The pretrained model used in this demo was developed by **Selahattin Serdar Helli** and **Andac Hamamci** at the Department of Biomedical Engineering, Faculty of Engineering, Yeditepe University, Istanbul, Turkey. The original training dataset was curated by Abdi et al. (2015).

---

## Author

**Victor Tablas**
- GitHub: [@vtablas001](https://github.com/vtablas001)
- Hugging Face: [@vtablas001](https://huggingface.co/vtablas001)

---

> *This application is intended for educational and demonstration purposes only. It is not a certified medical device and should not be used for clinical diagnosis without professional oversight.*
