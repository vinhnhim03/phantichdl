import pandas as pd
import numpy as np
import math
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import platform
import psutil
from tensorflow.python.keras.saving.saved_model.load import metrics



df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
print(df.head(10).to_string())# Tiền xử lý dữ liệu

# Tóm lược dữ liệu (Đo mức độ tập trung & mức độ phân tán)
description = df.describe()
mode = df.select_dtypes(include=['float64','int64']).mode().iloc[0]
mode.name = 'mode'
median = df.select_dtypes(include=['float64','int64']).median()
median.name = 'median'
description = description._append(mode)
description = description._append(median)
print(description)

# Kiểm tra data bị trùng
duplicated_rows_data = df.duplicated().sum()
print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
data = df.drop_duplicates()

# Kiểm tra tỷ lệ lỗi thiếu data
data_na = (df.isnull().sum() / len(df)) * 100
missing_data = pd.DataFrame({'Ty le thieu data': data_na})
print(missing_data)

# Quét qua các cột và đếm số lượng data riêng biệt
print("\nSO LUONG CAC DATA RIENG BIET:")

# Xem qua dataset
print(f"\n5 DONG DAU DATA SET:\n {data.head(5)}")

for column in data.columns:
    num_distinct_values = len(data[column].unique())
    print(f"{column}:{num_distinct_values} distinct values")


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

data_after_preprocessing = Z_score_normalize(df_cleaned)

df_numeric = df_cleaned.select_dtypes(include=['number'])

plt.figure(figsize=(15, 10))
for i, col in enumerate(df_numeric, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=df_numeric[col], color='skyblue')
    plt.title(f'Mức độ phân tán {col}')
    plt.ylabel(col)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Vẽ histogram cho các thuộc tính kiểu số
plt.figure(figsize=(15, 10))
for i, col in enumerate(df_numeric, 1):
    plt.subplot(4, 3, i)
    # Điều chỉnh bins dựa trên đặc điểm của từng cột
    if df_numeric[col].nunique() < 20:
        bins = df_numeric[col].nunique()
    elif df_numeric[col].max() - df_numeric[col].min() > 1000:
        bins = 50
    else:
        bins = 30
    sns.histplot(df_numeric[col], kde=True, bins=bins, color='skyblue')
    plt.title(f'Biểu đồ histogram cho cột {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

# Chuyển đổi các cột phân loại sang dạng số bằng Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ["holiday", "weather_main", ]

# Tính hệ số tương quan giữa các thuộc tính và giá trị mục tiêu traffic_volume
correlation_matrix = df_cleaned.corr()
traffic_correlation = correlation_matrix["traffic_volume"].sort_values(ascending=False)
print("\nTương quan của các thuộc tính với lưu lượng (traffic_volume):")
print(traffic_correlation)

# Vẽ biểu đồ scatter cho tất cả các thuộc tính với traffic_volume
plt.figure(figsize=(20, 15))
for i, col in enumerate(df_cleaned.columns[:-1], 1):
    plt.subplot(5, 3, i)
    sns.scatterplot(x=df_cleaned[col], y=df_cleaned["traffic_volume"], color='skyblue')
    plt.title(f'Mối quan hệ giữa {col} và traffic_volume')
    plt.xlabel(col)
    plt.ylabel('traffic_volume')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()

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
