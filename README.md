# Breast Cancer Detection Using EfficientNetB4

## 📌 Overview

Breast cancer is one of the most common and life-threatening diseases among women worldwide. Early and accurate diagnosis is important for effective treatment and improved patient outcomes.

This project presents a deep learning-based system for automated breast cancer classification using histopathological images. The proposed system uses an **EfficientNetB4-based deep learning model** to classify breast tissue images into two categories: **Benign** and **Malignant**.

The project also includes a Flask-based web application through which users can upload a histopathological image and obtain a prediction from the trained model.

---

## 🎯 Objectives

- Develop an automated system for breast cancer classification using histopathological images.
- Classify breast tissue images into **Benign** and **Malignant** categories.
- Apply image preprocessing techniques to prepare histopathological images for deep learning.
- Use transfer learning with **EfficientNetB4** for improved classification performance.
- Develop a simple web-based interface for uploading images and obtaining predictions.
- Reduce the dependency on time-consuming manual image analysis.

---

## 📂 Dataset

The project uses histopathological breast cancer images divided into two classes:

- **Benign**
- **Malignant**

The training dataset contains approximately:

- **2,480 Benign images**
- **5,429 Malignant images**

The complete dataset is not included in this repository because of its large size.

The dataset follows a class-based directory structure:

~~~text
dataset/
└── train/
    ├── benign/
    └── malignant/
~~~

---

## 🧠 Model Architecture

The project uses **EfficientNetB4** as the backbone architecture for breast cancer image classification.

Transfer learning is used to take advantage of features learned from a large-scale image dataset. The EfficientNetB4 model is adapted for the binary classification task of distinguishing between benign and malignant histopathological images.

The trained model is implemented using **TensorFlow/Keras**.

### Model Pipeline

~~~text
Histopathological Image
          ↓
Image Preprocessing
          ↓
EfficientNetB4
          ↓
Feature Extraction
          ↓
Classification Layer
          ↓
Benign / Malignant
~~~

---

## 🔄 Image Preprocessing

The input histopathological images undergo preprocessing before being passed to the deep learning model.

The preprocessing pipeline includes:

- Image loading
- Image resizing
- Pixel value normalization/preprocessing
- Preparation of images according to the EfficientNetB4 input requirements
- Conversion into a format suitable for model prediction

These preprocessing steps help provide consistent input to the deep learning model.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **EfficientNetB4**
- **Flask**
- **Jupyter Notebook**
- **HTML**
- **CSS**
- **NumPy**
- **OpenCV**
- **Matplotlib**

---

## 📁 Project Structure

~~~text
breast-cancer-detection-using-cnn/
│
├── static/
│   └── uploads/
│       └── uploaded images
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── projectbc.ipynb
├── projectbcfinal.html
└── README.md
~~~

---

## 🌐 Web Application

A Flask-based web application is included in the project.

The application allows a user to:

1. Open the breast cancer detection interface.
2. Upload a histopathological image.
3. Process the uploaded image.
4. Pass the image to the trained EfficientNetB4 model.
5. Generate a prediction.
6. Display the predicted class as **Benign** or **Malignant**.

---

## 🚀 Application Workflow

~~~text
User
  ↓
Upload Histopathological Image
  ↓
Flask Web Application
  ↓
Image Preprocessing
  ↓
Trained EfficientNetB4 Model
  ↓
Prediction
  ↓
Benign / Malignant Result
~~~

---

## 📊 Model Evaluation

The model can be evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

These evaluation metrics are useful for analyzing the classification performance of the proposed deep learning model.

---

## 📓 Project Files

### `projectbc.ipynb`

Contains the main deep learning workflow, including data processing, model development, training, evaluation, and experimentation.

### `app.py`

Contains the Flask application used to provide the web-based breast cancer prediction interface.

### `templates/`

Contains the HTML templates used by the Flask application.

### `static/uploads/`

Contains images used by the web application.

### `projectbcfinal.html`

Contains the exported HTML version of the project notebook/output.

---

## 🔮 Future Work

Future improvements can include:

- Increasing the diversity and size of the dataset.
- Improving model generalization on unseen histopathological images.
- Comparing EfficientNetB4 with other advanced CNN architectures.
- Implementing explainable AI techniques such as **Grad-CAM**.
- Deploying the application on a cloud platform.
- Improving the user interface and overall usability.
- Performing further hyperparameter optimization.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. The predictions generated by the system should not be considered a substitute for professional medical diagnosis.

---

## 👩‍💻 Project

**Breast Cancer Detection Using EfficientNetB4**

A deep learning-based research project for automated classification of breast cancer histopathological images into benign and malignant categories.
