# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\AIML_Internship\Task7\breast-cancer.csv")  # change filename if needed
print("Initial dataset shape:", df.shape)

# Step 2: Drop ID column (not useful for ML)
df.drop('id', axis=1, inplace=True)

# Step 3: Encode diagnosis column (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Step 4: Separate features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 5: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train SVM with Linear Kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)
y_pred_linear = svc_linear.predict(X_test)

print("\n--- Linear Kernel SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# Step 8: Train SVM with RBF Kernel
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)

print("\n--- RBF Kernel SVM ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# Step 9: Hyperparameter tuning using GridSearchCV for RBF Kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, verbose=1)
grid.fit(X_train, y_train)

print("\n--- Best Parameters for RBF Kernel ---")
print(grid.best_params_)
best_rbf = grid.best_estimator_
y_pred_best = best_rbf.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_best))

# Step 10: Cross-validation accuracy
cv_scores = cross_val_score(best_rbf, X_scaled, y, cv=5)
print("\nCross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Step 11: Visualize decision boundaries using PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

svc_vis = SVC(kernel='linear')
svc_vis.fit(X_pca, y)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='k')
plt.title("Decision Boundary Visualization (PCA reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
