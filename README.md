# Plant Disease Prediction using Convolutional Neural Networks 🌱

This project focuses on the detection and classification of plant diseases from leaf images using a custom-built Convolutional Neural Network (CNN) model. The model achieves an impressive **93% accuracy** on a dataset derived from the PlantVillage dataset, making it a reliable tool for assisting farmers and agricultural experts in diagnosing plant health issues.

---

## 📜 Project Overview

The goal of this project is to provide an efficient, accessible, and automated way to identify plant diseases based on leaf images. Leveraging deep learning techniques, the system performs multi-class classification to detect diseases such as rust, blight, and mildew in various crops.

---

## 🔧 Key Features

- **Custom CNN Architecture**: Built from scratch, the model comprises multiple convolutional layers for feature extraction and fully connected layers for classification.
- **Data Augmentation**: Enhanced the dataset with transformations like rotation, zooming, flipping, and shearing to improve generalization.
- **High Accuracy**: The model achieves a **93% validation accuracy**, ensuring robust performance on unseen data.
- **Visualization**: Training history and evaluation metrics are plotted for transparency and interpretability.
- **Scalable Predictions**: Includes a function to classify new images, making the model practical for real-world applications.

---

## 📚 Dataset

The model uses the **PlantVillage dataset**, a comprehensive collection of labeled plant leaf images, to train and evaluate the CNN model. The dataset includes a variety of crops with corresponding diseases such as **rust**, **blight**, **mildew**, and more. It is widely used for plant disease classification and offers a diverse set of images that help the model generalize well to new, unseen data.

- **Dataset source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- The dataset consists of:
  - **Healthy and diseased leaf images**
  - **Images across multiple plant species**

---

## 🛠️ Technologies Used

- **Python**: Programming language for implementation.
- **TensorFlow/Keras**: Framework for building and training the neural network.
- **NumPy & Matplotlib**: For data manipulation and visualization.
- **OpenCV & PIL**: For image preprocessing.
- **PlantVillage Dataset**: Dataset containing healthy and diseased leaf images.

---

## 📈 Model Performance

- **Training Accuracy**: 94%
- **Validation Accuracy**: 93%
![Stats](https://i.imgur.com/sDBKNmv.png "Stats Screenshot")

Loss curves and accuracy graphs are included to demonstrate the model’s learning process.

---

## 📂 Project Structure

```plaintext
Plant-Disease-Prediction/
├── main.py                 # Model training and evaluation script
├── class_indices.json      # Mapping of class indices to disease names
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
