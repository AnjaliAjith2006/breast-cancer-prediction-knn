# Step 1: Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
cancer = load_breast_cancer()

print(cancer.data.shape)
print(cancer.target_names)
print(cancer.feature_names)

# Step 3: Fetch features / independent variables
x = cancer.data

# Step 4: Fetch target value
y = cancer.target

# Step 5: Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Step 6 & 7: Create and train KNN classifier for K = 2 to 11
print("\nAccuracy for each K:")

for i in range(2, 12):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)

    # Step 8: Predict test data
    y_pred = knn.predict(x_test)

    # Step 9: Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("K =", i, "Accuracy =", accuracy)

# Step 10: Find optimal K using GridSearchCV

param_grid = {
    'n_neighbors': range(1, 32)
}
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
# Train GridSearchCV
grid_search.fit(x_train, y_train)

# Print best K and cross-validation score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Get best KNN model
best_knn = grid_search.best_estimator_

# Predict using optimal K
y_pred_best_k = best_knn.predict(x_test)

# Calculate test accuracy
accuracy_best_k = accuracy_score(y_test, y_pred_best_k)
print(
    "\nTest Accuracy with optimal K =",
    grid_search.best_params_['n_neighbors'],
    ":",
    round(accuracy_best_k, 4)
)
# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_k))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best_k))



