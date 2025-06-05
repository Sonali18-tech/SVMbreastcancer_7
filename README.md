 Support Vector Machines (SVM)

###  Objective

The goal of this task was to implement and understand **Support Vector Machines (SVM)** for **binary classification** using both **linear and non-linear kernels**, apply **hyperparameter tuning**, and visualize decision boundaries on 2D data using **PCA**.

---

###  What I Learned

* Working of SVMs and the concept of margin maximization.
* Differences between **Linear Kernel** and **RBF Kernel**.
* Role of important hyperparameters like `C` and `gamma`.
* Visualizing high-dimensional data using **PCA (Principal Component Analysis)**.
* Performing **Grid Search** and **Cross-Validation** to optimize models.

---

### Dataset Used

**Breast Cancer Wisconsin (Diagnostic) Dataset**

* Source: [Kaggle - yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
* Total Rows: 569
* Columns: 32
* Target Variable: `diagnosis` (Malignant = 1, Benign = 0)

### 🛠 Libraries Used

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

###  Steps Performed

#### 1. **Dataset Preprocessing**

* Loaded the dataset using `pandas`.
* Removed the `id` column as it does not contribute to learning.
* Mapped the `diagnosis` column: `M` → 1 (Malignant), `B` → 0 (Benign).
* Standardized the feature set using `StandardScaler` to ensure SVM performs effectively.

#### 2. **Train-Test Split**

* Split the data into training (80%) and testing (20%) sets using `train_test_split`.

#### 3. **Model Training**

* Trained two separate SVM classifiers:

  * **Linear Kernel SVM**
  * **RBF (Radial Basis Function) Kernel SVM**

#### 4. **Model Evaluation**

* Evaluated both models using:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report

#### 5. **Hyperparameter Tuning**

* Used `GridSearchCV` to tune `C` and `gamma` parameters for the RBF kernel.
* Found the best combination for optimized accuracy.

#### 6. **Cross-Validation**

* Performed 5-fold cross-validation to assess model generalization.

#### 7. **Visualization**

* Reduced dimensionality using PCA to 2 components.
* Plotted the decision boundary on the 2D PCA projection for visual insight.

---

###  Results

| Model            | Accuracy | Best Parameters      |
| ---------------- | -------- | -------------------- |
| SVM (Linear)     | \~96%    | Default              |
| SVM (RBF)        | \~98%    | `C=10`, `gamma=0.01` |
| Cross-Validation | \~97.8%  | -                    |


Author: Sonali18-tech
