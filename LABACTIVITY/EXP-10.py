import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist
# Different feature dimensions
dimensions = [2, 5, 10, 20, 50]
print("Dimension\tAvg Distance\tAccuracy")
for d in dimensions:
# Generate Dataset
X, y = make_classification(
n_samples=500,
n_features=d,
n_informative=max(2, d//2),
n_redundant=0,
random_state=42
)
# Average Euclidean Distance
avg_distance = np.mean(pdist(X))
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(

X, y,
test_size=0.2,
random_state=42
)
# KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"{d}\t\t{avg_distance:.2f}\t\t{accuracy:.2f}")
