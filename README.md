# Fruit_360

# Image Classification using Logistic Regression and K-Means  
### Fruits-360 Dataset

## 📖 Project Overview
This project explores the application of **classical machine learning algorithms** on image data using the **Fruits-360 dataset**.  
Instead of relying on deep learning models, the project demonstrates how traditional algorithms can achieve strong performance when combined with proper preprocessing and dimensionality reduction techniques.

The project compares:
- **Logistic Regression** (Supervised Learning)
- **K-Means Clustering** (Unsupervised Learning)

---

## 🎯 Objectives
- Convert image data into numerical feature vectors suitable for classical ML models.
- Apply **Principal Component Analysis (PCA)** to reduce dimensionality.
- Train and evaluate a **Logistic Regression** classifier.
- Apply **K-Means** to explore the intrinsic structure of the data.
- Compare supervised and unsupervised learning approaches on the same dataset.

---

## 📂 Dataset
- **Name:** Fruits-360  
- **Image Size:** 100 × 100 pixels  
- **Background:** Plain, controlled environment  
- **Classes Used:** 5 fruit categories  

The dataset used in this project is **Fruits-360**.

Due to its large size, the dataset is **not included** in this repository.  
It can be downloaded from Kaggle using the following link:

https://www.kaggle.com/datasets/moltean/fruits

The controlled nature of the dataset makes it suitable for evaluating classical machine learning algorithms.

---

## 🛠️ Preprocessing Steps
1. Load images using OpenCV.
2. Convert images to grayscale.
3. Resize all images to **100 × 100 pixels**.
4. Flatten images into **1D vectors (10,000 features)**.
5. Normalize pixel values to the range **[0, 1]**.
6. Split data into training (80%) and testing (20%) sets using **stratified sampling**.

---

## 📉 Dimensionality Reduction (PCA)
- **Method:** Principal Component Analysis (PCA)
- **Variance Retained:** 95%
- **Feature Reduction:**  
  - Before PCA: **10,000 features**  
  - After PCA: **137 features**

### Why PCA?
- Reduces computational complexity  
- Removes redundant and noisy features  
- Mitigates the curse of dimensionality  
- Improves model generalization  

---

## 🤖 Models Used

### 1️⃣ Logistic Regression (Supervised Learning)
- Trained on PCA-reduced features
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

**Results:**
- Accuracy ≈ **99.8%**
- Only **one misclassification** on the test set

---

### 2️⃣ K-Means Clustering (Unsupervised Learning)
- Applied to PCA-transformed features
- Number of clusters: **5**
- Evaluation metric:
  - Silhouette Score

**Results:**
- Silhouette Score ≈ **0.28**

This score indicates **moderate cluster separation**, which is expected for image data due to visual similarities between different fruit classes.

---

## 📊 Comparison

| Aspect | Logistic Regression | K-Means |
|------|--------------------|--------|
| Learning Type | Supervised | Unsupervised |
| Uses Labels | Yes | No |
| Task | Classification | Clustering |
| Performance | Very High | Moderate |
| Purpose | Accurate prediction | Data exploration |

---

## 📚 Libraries Used

The following Python libraries were used in this project:

- **OpenCV (`cv2`)**  
  Used for image processing tasks such as reading images, converting them to grayscale, and resizing them.

- **OS (`os`)**  
  Used for file and directory handling and navigating dataset folders.

- **NumPy (`numpy`)**  
  Used for numerical operations and efficient handling of image data.

- **Matplotlib (`matplotlib.pyplot`)**  
  Used for visualizing images, clusters, and model results.

- **Scikit-learn (`sklearn`)**  
  Used for machine learning tasks, including:
  - `train_test_split`
  - `StandardScaler`
  - `PCA`
  - `LogisticRegression`
  - `KMeans`
  - `accuracy_score`, `classification_report`, `confusion_matrix`

---

## ⚠️ Limitations
- Flattened pixel features do not capture spatial relationships.
- Dataset images are captured under controlled conditions.
- Grayscale conversion removes color information that may improve classification for some fruits.

---

## 🚀 Future Work
- Use color-based or handcrafted image features.
- Apply Convolutional Neural Networks (CNNs).
- Test the model on real-world images with complex backgrounds.
- Explore alternative dimensionality reduction techniques.

---

## 📁 Project Structure
