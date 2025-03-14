import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
# Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance/data
# %%
data = pd.read_csv(r"C:\Users\PMLS\Desktop\SEM-6\Data Mining And Machine Learning\Data-Mining-and-Machine-Learning\Assignment-5\internet.csv")

# %%
data.shape

# %%
data.head()

# %%
data.info()
#%%
data.describe()

# %%
print("\nMissing values in the dataset:")
data.isnull().sum()

# %%
categoricals = data.select_dtypes(include=['object']).columns.tolist()
numericals = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
categoricals, numericals
#%%
data.shape
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # 25th percentile
        Q3 = df[col].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


data = remove_outliers_iqr(data, [each for each in numericals])
data.shape

# %%
data.head()
# %%
X = data.drop(columns=['Internet_speed']) # All columns except Target column
y = data['Internet_speed'] # Target column
numericals.remove('Internet_speed')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train[numericals] = scaler.fit_transform(X_train[numericals])
X_test[numericals]  = scaler.transform(X_test[numericals])

#%%
dt_model = DecisionTreeRegressor(max_depth=10, random_state=62)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
#%%
rf_model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=33)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

#%%
svm_model = SVR(kernel='rbf', C=100, gamma=0.1,epsilon=0.01)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

#%%
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Performance:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n")

#%%

evaluate_model("Decision Tree", y_test, dt_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("SVM", y_test, svm_pred)

#%%
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
dt_pred_sorted = dt_pred[sorted_idx]
rf_pred_sorted = rf_pred[sorted_idx]
svm_pred_sorted = svm_pred[sorted_idx]

#%%
plt.figure(figsize=(10, 6))
plt.plot(y_test_sorted, label="Actual Internet Speed", color="black", linestyle="-", linewidth=2)
plt.plot(dt_pred_sorted, label="Decision Tree", color="blue", linestyle="--")
plt.plot(rf_pred_sorted, label="Random Forest", color="red", linestyle=":")
plt.plot(svm_pred_sorted, label="SVM", color="green", linestyle="-." )

plt.xlabel("Test Samples (Sorted)")
plt.ylabel("Internet Speed")
plt.title("Actual vs Predicted Internet Speed for Decision Tree, Random Forest & SVM")
plt.legend()
plt.show()