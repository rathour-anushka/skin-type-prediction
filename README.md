# 🧴 Skin Type Prediction

A lightweight deep learning app that predicts skin type from facial images using MobileNetV2. Designed for fast, user-friendly skincare insights.

## 🚀 Features

- Predicts skin type: **Oily**, **Dry**, **Combination**, **Sensitive**, or **Normal**
- Built with `MobileNetV2` for fast inference and mobile compatibility
- Interactive UI via Streamlit
- Modular folder structure for maintainability
- Optional Grad-CAM visualizations for explainability

## 🧠 Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Input shape**: (224, 224, 3)
- **Classifier head**: GlobalAveragePooling → Dense → Softmax
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam`

## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rathour-anushka/skin-type-prediction.git
   cd skin-type-prediction

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
