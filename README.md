# Neural Networks and Deep Learning
Assignments for the "Neural Networks and Deep Learning" Course Faculty of Engineering, AUTh School of Informatics: Neural Network Architectures and Hybrid Models with Custom Implementations (SVM, MLP, and RBF NNs from scratch, CNNs, Autoencoders)

---

### Assignments for "Neural Networks and Deep Learning" Coursework (2023)
Assignment for the "Neural Networks and Deep Learning" Course  
Faculty of Engineering, AUTh  
School of Electrical and Computer Engineering  
Electronics and Computers Department

📚 *Course:* Computer Graphics                   
🏛️ *Faculty:* AUTh - School of Electrical and Computer Engineering  
📅 *Semester:* 9th Semester, 2023–2024


---

# 🤖 Neural Networks & Deep Learning – Assignment 1
### 📌 Title: Image Classification using Neural Networks on CIFAR-10 Dataset


## 📖 Objective

This assignment focuses on **image classification** using the **CIFAR-10 dataset**, exploring various machine learning and deep learning models including:

- 🔎 **KNN & Nearest Centroid classifiers**
- 🔬 **Multi-Layer Perceptron (MLP)**
- 🧠 **Convolutional Neural Networks (CNN)**

The goal is to progressively enhance model performance, optimize hyperparameters, and address overfitting using techniques like **Dropout**, **Batch Normalization**, and **Early Stopping**.


## 📂 Dataset

- **CIFAR-10:** 60,000 color images (32x32x3) in 10 categories:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Train/Test Split:** 50,000 training images / 10,000 testing images
- **Preprocessing:**
  - Normalization: [0, 1]
  - Flattening images for MLP
  - One-hot encoding of labels
  - PCA (Principal Component Analysis): Dimensionality reduced to 99 features (retaining 90% variance)



## 🔍 Models & Methods

### 1️⃣ **KNN & Nearest Centroid Classifiers**

- **Variants:**
  - KNN (k=1, k=3)
  - Nearest Centroid (Euclidean & Manhattan distances)
- **Observation:**  
  Poor accuracy; limited by their inability to capture semantic features in images.


### 2️⃣ **Multi-Layer Perceptron (Fully Connected)**

- **Architecture:**
  - Input: Flattened image (3072 features, or 99 after PCA)
  - Hidden Layers: 1–3 layers, with 512 or 256 neurons
  - Output: 10 neurons with **Softmax** activation
- **Activation Functions:**
  - Hidden: **ReLU**
  - Output: **Softmax**
- **Loss Function:**
  - Categorical Cross-Entropy:  
    `Loss = - Σ y_true * log(y_pred)`
- **Optimizer:** **Adam**
- **Batch Size:** 128
- **Evaluation Metrics:** Accuracy & F1 Score

- **Experiments:**
  - Varying **learning rates** (0.001, 0.01)
  - Optimizers (Adam vs SGD)
  - Adding layers (tested up to 3 layers)
  - **PCA impact:** Reduced training time with minimal performance loss
  - Regularization:
    - L1 Regularization (λ = 0.01)
    - Dropout (p = 0.4)
    - Batch Normalization
    - Early Stopping (patience = 10)

- **Key Findings:**
  - **Optimal MLP:** 2 hidden layers (512 neurons), dropout 0.4, learning rate 0.001
  - Peak accuracy: ~58%
  - **Dropout & Batch Normalization:** Crucial for mitigating overfitting
  - **Hyperparameter Tuning:** Used `KerasTuner` with RandomSearch to optimize neurons, dropout rate, and learning rate


### 3️⃣ **Convolutional Neural Networks (CNN)**

- **Why CNN:**  
  Outperforms MLP by capturing spatial hierarchies and patterns (translation invariance)

- **Architectures Tested:**
  
  **Model 1:**
  - Conv Layer (32 filters, 4x4)
  - MaxPooling (4x4)
  - Conv Layer (64 filters, 8x8)
  - MaxPooling (4x4)
  - Conv Layer (128 filters, 8x8)
  - MaxPooling (4x4)
  - Flatten → Dense (512) → Output (10)

  **Model 2:**
  - Adds extra Conv layers (16 & 128 filters), reduces pooling size (2x2)

  **Model 3:**
  - Double Conv layers at each stage
  - Smaller kernel (2x2)
  - MaxPooling (2x2)

- **Regularization:**
  - Dropout (p = 0.2 → 0.3 for best results)

- **Results:**
  - CNN achieved ~78–80% accuracy
  - **Best Model:** Model 3 with 0.3 dropout after tuning
  - CNNs demonstrated much stronger performance than MLP (confirming their effectiveness for image data)


## 📊 Summary of Results

| Model                  | Best Accuracy |
|------------------------|---------------|
| KNN / Nearest Centroid | ~20%          |
| MLP (Optimal)          | ~58%          |
| CNN (Best)             | ~80%          |



## 📝 Key Takeaways

- 🔧 **MLPs** are limited for image data without feature engineering or CNN-like architectures.
- 🧠 **CNNs** leverage convolution to efficiently learn spatial hierarchies and generalize better.
- 🎯 **Regularization (Dropout, L1), Batch Normalization, and Early Stopping** play a critical role in improving generalization and preventing overfitting.
- ⚙️ **Hyperparameter tuning** significantly impacts performance.


---

# 🤖 Neural Networks & Deep Learning – Assignment 2
### 📌 Title: Binary Classification using Support Vector Machines (SVM)

---

## 📖 Objective

This assignment focuses on implementing and tuning **Support Vector Machines (SVMs)** for **binary classification**, using both **image data (CIFAR-10)** and a **tabular dataset (Breast Cancer Wisconsin)**.

The key goals:
- Build an **SVM class** from scratch (QP optimization with `cvxopt`)
- Test with different **kernels**: Linear, RBF (Gaussian), Polynomial, and Sigmoid
- Tune **hyperparameters** using grid search and **cross-validation**
- Analyze and interpret **classification reports & confusion matrices**


## 📂 Datasets

1️⃣ **CIFAR-10:**
- 60,000 RGB images (32×32×3), 10 categories.
- For SVM: selected **2 classes** → new binary dataset:  
  - Train: 10,000 samples  
  - Test: 2,000 samples  
- Preprocessing:
  - Normalization [0, 1]
  - Reshape to 1D arrays
  - Labels: {-1, 1}

2️⃣ **Breast Cancer Wisconsin:**
- 569 samples × 30 features
- Labels:  
  - Malignant (M) → -1  
  - Benign (B) → 1
- Preprocessing:
  - Normalization [0, 1]
  - PCA: Reduced features to retain 95% variance


## 🔍 Methods

### 🔗 **Nearest Centroid & K-Nearest Neighbor (KNN)**

- **KNN (k=1, 3)**
- **Nearest Centroid:**
  - Euclidean distance
  - Manhattan distance

➡️ **Results (Breast Cancer Dataset):**
- Nearest Centroid (Euclidean): ✅ Best performance (~93%)
- 1-NN: Slightly worse (~83.5%)


### 🛠️ **SVM Implementation**

Custom SVM built with:
- **QP Solver:** `cvxopt`
- **Kernels:**
  - Linear
  - RBF (Gaussian)
  - Polynomial
  - Sigmoid

#### **Key Parameters:**
- `C`: Regularization (penalty for misclassification)
- `gamma`: RBF kernel parameter
- `degree`: Polynomial kernel degree
- `constant`: Polynomial constant term


## 🧠 Kernels Overview

| Kernel               | Formula (simplified)                                         |
|----------------------|--------------------------------------------------------------|
| Linear               | \( K(x, y) = x \cdot y \)                                    |
| Gaussian (RBF)       | \( K(x, y) = \exp(-\gamma \|x - y\|^2) \)                   |
| Polynomial           | \( K(x, y) = (x \cdot y + c)^{d} \)                          |
| Sigmoid              | \( K(x, y) = \tanh(\alpha x \cdot y + c) \)                  |



## 🔎 Results

### 1️⃣ **Linear Kernel (C = 1)**

- ✅ Malignant: 42/43 correct
- ✅ Benign: 60/71 correct
- Confusion matrix:  
  Very strong performance, with minor misclassification of Benign as Malignant.


### 2️⃣ **Gaussian (RBF) Kernel (C = 1, gamma = 0.1)**

- ✅ Malignant: 37/43 correct
- ✅ Benign: 71/71 correct
- Confusion matrix:  
  High accuracy but 6 false negatives (missed cancer diagnosis).



### 3️⃣ **Polynomial Kernel (C = 1, degree = 2)**

- ✅ Malignant: 41/43 correct
- ✅ Benign: 66/71 correct
- Confusion matrix:  
  Slightly better balance than RBF, higher training time.


## 🛠️ Hyperparameter Tuning

- **Grid Search + 5-Fold Cross Validation**
- Dataset: Breast Cancer Wisconsin

### A️⃣ Linear Kernel

| C        | Accuracy |
|----------|----------|
| 0.1      | 0%       |
| 1        | 89.45%   |
| 10       | 90.69%   |
| 100      | 91.57%   |
| 1000     | 92.45%   |
| 10e3     | 91.45%   |

### B️⃣ RBF Kernel

| C    | Gamma | Accuracy |
|-------|--------|----------|
| 1     | 0.1    | 86.99%   |
| 10    | 0.1    | 88.23%   |
| 100   | 0.01   | 88.22%   |
| 1000  | 1      | 65.37%   |
| 1000  | 100    | 80.66%   |

Best: ~91.56% (C=100, gamma=0.01)

### C️⃣ Polynomial Kernel

| C    | Degree | Constant | Accuracy |
|-------|--------|----------|----------|
| 1     | 3      | 2        | 90.51%   |
| 10    | 4      | 2        | 92.44%   |
| 100   | 2      | 2        | 92.61%   |
| 1000  | 3      | 1        | 92.97%   |


## 📊 Summary Table

| Model                                  | Accuracy     |
|----------------------------------------|--------------|
| 1-NN                                   | 83.55%       |
| 3-NN                                   | 91.35%       |
| Nearest Centroid (Euclidean)           | 93.04%       |
| SVM (Linear Kernel)                    | 92.45%       |
| SVM (RBF Kernel)                       | 91.56%       |
| SVM (Polynomial Kernel)                | 92.97%       |


## 📝 Observations

- 🔑 **Polynomial Kernel** delivered the **best overall results** (~92.97%) but was the slowest to train.
- ✅ **Nearest Centroid** surprisingly performed very well (~93%).
- ⚠️ **RBF Kernel:** Despite high accuracy, problematic false negatives (missing malignant cases).
- 📊 **Hyperparameter tuning** was crucial, especially for C & gamma in RBF, and degree/constant in Polynomial kernels.
- 🤖 For small datasets, **SVMs with well-tuned hyperparameters** remain highly effective.

---

# 🤖 Neural Networks & Deep Learning – Assignment 3

### 📌 Title: Multiclass Classification using RBF Neural Network (CIFAR-10)

📅 Semester: 2023–2024  
🏛️ Institution: AUTh – School of Electrical and Computer Engineering  
👨‍💻 Student: Ioannis Deirmentzoglou (AEM: 10015)

---

## 📖 Objective

In this assignment, a **Radial Basis Function Neural Network (RBF NN)** was implemented to tackle a **multiclass classification** problem using the **CIFAR-10 dataset**.

The goals:
- Design an RBF network for **10-class classification**
- Visualize data via **PCA (2D and 3D)**
- Benchmark RBF NN vs. **KNN** and **Nearest Centroid**
- Experiment with different **number of centers** and training strategies

---

## 📂 Dataset

**CIFAR-10:**
- 60,000 images (32×32×3), 10 categories:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Train Set:** 50,000 samples
- **Test Set:** 10,000 samples
- Preprocessing:
  - Normalization [0, 1]
  - Flatten images to 1D vectors (3072 features)
  - **One-hot encoding** on labels

---

## 🧪 Preprocessing & PCA

- **PCA Reduction:**  
  - Retain **90% or 95% variance**
  - ~217 features needed for 95% variance

- **Visualization:**
  - **2D PCA Scatter Plot** (First 2 principal components)
  - **3D PCA Scatter Plot** (First 3 principal components)

📝 **Observation:**  
Classes are **not linearly separable** in 2D/3D → High-dimensional features required for separation.

---

## 🔍 Baseline Classifiers

### 📈 KNN & Nearest Centroid

- **KNN (k = 1, 3)**
- **Nearest Centroid:**
  - Euclidean distance
  - Manhattan distance

➡️ **Findings:**
- Overall **low accuracy**
- Nearest Centroid:
  - Better than KNN
  - Faster testing time
- CIFAR-10 **challenging for simple classifiers**

---

## 🧠 RBF Neural Network Architecture

| Layer            | Description                                                        |
|------------------|--------------------------------------------------------------------|
| Input Layer      | Raw image vectors (3072 features) or PCA-reduced features          |
| Hidden Layer     | RBF nodes (Gaussian functions), centers found via **KMeans**       |
| Output Layer     | Linear activation (multiclass output with softmax-like behavior)   |

---

## ⚙️ Key Functions

- `rbfKernel()`:  
  Computes the **RBF kernel matrix** between data points and centers.

- `rbfKernelDerivative()`:  
  Computes the derivative of the RBF kernel (for backprop).

- `f1Score()`:  
  Calculates the F1 score metric.

- `MSE() / MSEDerivative()`:  
  Standard loss & gradient calculations.

---

## 🛠️ Training Procedure

1. **Initialize centers:**
   - **KMeans clustering** on training data
   - Common formula for spread:  
   \[
   \sigma = \frac{\text{maxDistance}}{\sqrt{2 \times \text{numCenters}}}
   \]

2. **Initial Weights:**
   - Using **pseudo-inverse** of RBF kernel output.

3. **Optional Fine-tuning:**
   - Backpropagation with:
     - Epochs: 20
     - Learning rate: 0.01

---

## 📊 Experiments & Results

| Num Centers | PCA Features | Training | Accuracy (Test Set) | Notes                                       |
|-------------|--------------|----------|---------------------|---------------------------------------------|
| 217         | 217          | Yes      | ~ Low (~10%)        | Very long training (~7 minutes total)       |
| 10          | Full         | No       | Low                 | Fast but poor accuracy                      |
| 20          | Full         | No       | Low                 | Slightly better than 10 centers             |
| 100         | Full         | No       | Moderate            | Accuracy improves                           |
| 1000        | Full         | No       | ~25%                | Best performance but **still poor overall** |

---

### 📈 Training Insights

- **Epoch training:**  
  Only minor improvement; sometimes worsened results.

- **Confusion Matrix:**  
  - Ship & Deer: ✅ Best recognized
  - Car: ❌ Worst performance

- **Sample Predictions:**  
  - Visualizations of **correct** vs. **incorrect** predictions

- **Time:**  
  - Larger center counts (e.g., 1000) → **much longer training time**

---

## 🔎 Observations

- 🚩 **Challenge:** CIFAR-10 is **high-dimensional & complex**.
- ❗ RBF NN **did not scale well** even with many centers.
- ✅ Nearest Centroid & RBF NN both outperformed KNN but **failed to deliver strong accuracy**.
- 📉 Increasing centers boosts accuracy **slightly** but at a significant **computational cost**.
- 🧐 **Conclusion:** RBF NNs are not well-suited for CIFAR-10 **without further enhancements** (e.g., CNNs perform much better).

---

# 🤖 Neural Networks & Deep Learning – Assignment 3
### 📌 Title: Multiclass Classification using RBF Neural Network (CIFAR-10)


---

## 📖 Objective

In this assignment, a **Radial Basis Function Neural Network (RBF NN)** was implemented to tackle a **multiclass classification** problem using the **CIFAR-10 dataset**.

The goals:
- Design an RBF network for **10-class classification**
- Visualize data via **PCA (2D and 3D)**
- Benchmark RBF NN vs. **KNN** and **Nearest Centroid**
- Experiment with different **number of centers** and training strategies



## 📂 Dataset

**CIFAR-10:**
- 60,000 images (32×32×3), 10 categories:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Train Set:** 50,000 samples
- **Test Set:** 10,000 samples
- Preprocessing:
  - Normalization [0, 1]
  - Flatten images to 1D vectors (3072 features)
  - **One-hot encoding** on labels



## 🧪 Preprocessing & PCA

- **PCA Reduction:**  
  - Retain **90% or 95% variance**
  - ~217 features needed for 95% variance

- **Visualization:**
  - **2D PCA Scatter Plot** (First 2 principal components)
  - **3D PCA Scatter Plot** (First 3 principal components)

📝 **Observation:**  
Classes are **not linearly separable** in 2D/3D → High-dimensional features required for separation.



## 🔍 Baseline Classifiers

### 📈 KNN & Nearest Centroid

- **KNN (k = 1, 3)**
- **Nearest Centroid:**
  - Euclidean distance
  - Manhattan distance

➡️ **Findings:**
- Overall **low accuracy**
- Nearest Centroid:
  - Better than KNN
  - Faster testing time
- CIFAR-10 **challenging for simple classifiers**



## 🧠 RBF Neural Network Architecture

| Layer            | Description                                                        |
|------------------|--------------------------------------------------------------------|
| Input Layer      | Raw image vectors (3072 features) or PCA-reduced features          |
| Hidden Layer     | RBF nodes (Gaussian functions), centers found via **KMeans**       |
| Output Layer     | Linear activation (multiclass output with softmax-like behavior)   |



## ⚙️ Key Functions

- `rbfKernel()`:  
  Computes the **RBF kernel matrix** between data points and centers.

- `rbfKernelDerivative()`:  
  Computes the derivative of the RBF kernel (for backprop).

- `f1Score()`:  
  Calculates the F1 score metric.

- `MSE() / MSEDerivative()`:  
  Standard loss & gradient calculations.



## 🛠️ Training Procedure

1. **Initialize centers:**
   - **KMeans clustering** on training data
   - Common formula for spread:  
   \[
   \sigma = \frac{\text{maxDistance}}{\sqrt{2 \times \text{numCenters}}}
   \]

2. **Initial Weights:**
   - Using **pseudo-inverse** of RBF kernel output.

3. **Optional Fine-tuning:**
   - Backpropagation with:
     - Epochs: 20
     - Learning rate: 0.01



## 📊 Experiments & Results

| Num Centers | PCA Features | Training | Accuracy (Test Set) | Notes                                       |
|-------------|--------------|----------|---------------------|---------------------------------------------|
| 217         | 217          | Yes      | ~ Low (~10%)        | Very long training (~7 minutes total)       |
| 10          | Full         | No       | Low                 | Fast but poor accuracy                      |
| 20          | Full         | No       | Low                 | Slightly better than 10 centers             |
| 100         | Full         | No       | Moderate            | Accuracy improves                           |
| 1000        | Full         | No       | ~25%                | Best performance but **still poor overall** |


### 📈 Training Insights

- **Epoch training:**  
  Only minor improvement; sometimes worsened results.

- **Confusion Matrix:**  
  - Ship & Deer: ✅ Best recognized
  - Car: ❌ Worst performance

- **Sample Predictions:**  
  - Visualizations of **correct** vs. **incorrect** predictions

- **Time:**  
  - Larger center counts (e.g., 1000) → **much longer training time**


## 🔎 Observations

- 🚩 **Challenge:** CIFAR-10 is **high-dimensional & complex**.
- ❗ RBF NN **did not scale well** even with many centers.
- ✅ Nearest Centroid & RBF NN both outperformed KNN but **failed to deliver strong accuracy**.
- 📉 Increasing centers boosts accuracy **slightly** but at a significant **computational cost**.
- 🧐 **Conclusion:** RBF NNs are not well-suited for CIFAR-10 **without further enhancements** (e.g., CNNs perform much better).




---


## 🤝 Contributor

- **Ioannis Deirmentzoglou** – [GitHub Profile](https://github.com/jonnyderme)

---
