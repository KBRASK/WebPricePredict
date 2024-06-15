import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 加载训练数据
train_data = pd.read_csv('train_data/train.csv')

# 去掉 'Id' 列
train_data = train_data.drop(columns=['Id'])

# 分离特征和目标变量
X_train = train_data.drop(columns=['SalePrice'])
y_train = train_data['SalePrice']

# 定义数值型和类别型特征
num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

# 定义数值型和类别型数据的处理管道
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 使用 ColumnTransformer 来应用上述管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 处理训练数据
X_train_transformed = preprocessor.fit_transform(X_train)

# 获取数值型特征的平均值
num_means = preprocessor.named_transformers_['num'].named_steps['imputer'].statistics_

# 获取类别型特征的最频繁值
cat_modes = preprocessor.named_transformers_['cat'].named_steps['imputer'].statistics_

print("Numeric Features Means:")
for name, mean in zip(num_features, num_means):
    print(f"{name}: {mean}")

print("\nCategorical Features Most Frequent Values:")
for name, mode in zip(cat_features, cat_modes):
    print(f"{name}: {mode}")

for name in X_train:
    print(name)
