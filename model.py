import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv("Iris.csv")
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Huấn luyện mô hình KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 4. Lưu mô hình đã huấn luyện
joblib.dump(knn, "knn_model.pkl")
print("✅ Mô hình KNN đã được huấn luyện và lưu vào knn_model.pkl")
