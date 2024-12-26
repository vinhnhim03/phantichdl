import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV và tạo DataFrame
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv", delimiter=",")  # đọc dữ liệu từ tệp

# Tiền xử lý dữ liệu
def handel_missing_value(df):
    df_miss = df.columns[df.isna().sum() > 1]
    print(df_miss)
    for i in df_miss:
        print(i)
    df_numeric = df.select_dtypes(include=['number'])  # Chọn các cột dữ liệu số
    print(df_numeric.to_string())
    df_cleaned = df_numeric.dropna()  # Xóa các dòng có giá trị khuyết
    return df_cleaned


df_cleaned = handel_missing_value(df)

# Chuẩn hóa theo Z_score normalization
def Z_score_normalize(df_cleaned):
    mean = df_cleaned.mean()
    std_dev = df_cleaned.std()

    normalized_data = (df_cleaned - mean) / std_dev
    return normalized_data

# Chuẩn hóa dữ liệu
data_after_preprocessing = Z_score_normalize(df_cleaned)

# Hàm hồi quy đơn biến
def linear_regression_model_univariate(data_after_preprocessing):
    X = data_after_preprocessing[['temp']]  # Biến 'temp' làm biến độc lập
    y = data_after_preprocessing['traffic_volume']  # Biến phụ thuộc

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)  # Huấn luyện mô hình

    y_pred = model.predict(X_test)  # Dự đoán kết quả

    # Đánh giá mô hình
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = round(model.score(X_test, y_test), 2)

    print("Đánh giá mô hình hồi quy đơn biến:")
    print("RMSE:", rmse)
    print("R2:", r2)

    # Lưu dự đoán và so sánh với giá trị thực tế
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.to_csv("univariate_predictions.csv", index=False)  # Lưu kết quả vào tệp CSV

    # Hiển thị đồ thị
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red')
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume (Univariate)')
    plt.show()

    return model, rmse, r2

# Hàm hồi quy đa biến
def linear_regression_model_multivariate(data_after_preprocessing):
    X = data_after_preprocessing[['rain_1h', 'snow_1h', 'clouds_all']]  # Các biến độc lập
    y = data_after_preprocessing['traffic_volume']  # Biến phụ thuộc

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)  # Huấn luyện mô hình

    y_pred = model.predict(X_test)  # Dự đoán kết quả

    # Đánh giá mô hình
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = round(model.score(X_test, y_test), 2)

    print("Đánh giá mô hình hồi quy đa biến:")
    print("RMSE:", rmse)
    print("R2:", r2)

    # Lưu dự đoán và so sánh với giá trị thực tế
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.to_csv("multivariate_predictions.csv", index=False)  # Lưu kết quả vào tệp CSV

    # Hiển thị đồ thị
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red')
    plt.xlabel('Actual Traffic Volume')
    plt.ylabel('Predicted Traffic Volume')
    plt.title('Actual vs Predicted Traffic Volume (Multivariate)')
    plt.show()

    return model, rmse, r2


# Sử dụng mô hình hồi quy đơn biến và đa biến
model_univariate, rmse_univariate, r2_univariate = linear_regression_model_univariate(data_after_preprocessing)
model_multivariate, rmse_multivariate, r2_multivariate = linear_regression_model_multivariate(data_after_preprocessing)

# In kết quả của cả hai mô hình
print("Kết quả hồi quy đơn biến:")
print("RMSE:", rmse_univariate)
print("R2:", r2_univariate)

print("\nKết quả hồi quy đa biến:")
print("RMSE:", rmse_multivariate)
print("R2:", r2_multivariate)
