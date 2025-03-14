# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels import api as sm
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
# Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance/data
# %%
data = pd.read_csv("insurance.csv")

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


data = remove_outliers_iqr(data, ['age', 'children','bmi'])
data.shape
# %%
for each in categoricals:
    print(each, len(data[each].unique().tolist()))

# %%
threshold = 25
data[categoricals] = data[categoricals].apply(
    lambda each: each.where(each.isin(each.value_counts().nlargest(threshold).index), "Other"))

# %%
data.head()


#%%
encoder = LabelEncoder()
for each in categoricals:
    data[each] = encoder.fit_transform(data[each])


#%%
data.head()

# %%
X = data.drop(columns=['charges']) # All columns except Target column
y = data['charges'] # Target column
numericals.remove('charges')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train[numericals] = scaler.fit_transform(X_train[numericals])
X_test[numericals]  = scaler.transform(X_test[numericals])

#%%
from sklearn.tree import DecisionTreeRegressor


#%%
model = DecisionTreeRegressor( max_depth=5,random_state=62)
model.fit(X_train, y_train)
#%%

print('DecisionTreeRegressor Train Score is : ' , model.score(X_train, y_train))
print('DecisionTreeRegressor Test Score is : ' , model.score(X_test, y_test))
print('-'*70)

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_test)
dt_pred = y_pred.copy()
mse = mean_squared_error(y_test, y_pred,multioutput='uniform_average')
print(f"Mean Squared Error (MSE): {mse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")

#%%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=250,max_depth=5, random_state=33)
model.fit(X_train, y_train)
print('Random Forest Regressor Train Score is : ' , model.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , model.score(X_test, y_test))
print('-'*70)
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_test)
rf_pred = y_pred.copy()
mse = mean_squared_error(y_test, y_pred,multioutput='uniform_average')
print(f"Mean Squared Error (MSE): {mse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2}")

#%%
import matplotlib.pyplot as plt
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
dt_pred_sorted = dt_pred[sorted_idx]
rf_pred_sorted = rf_pred[sorted_idx]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sorted, label="Actual Charges", color="black", linestyle="-", linewidth=2)
plt.plot(dt_pred_sorted, label="Decision Tree Predictions", color="blue", linestyle="--")
plt.plot(rf_pred_sorted, label="Random Forest Predictions", color="red", linestyle=":")

plt.xlabel("Test Samples (Sorted)")
plt.ylabel("Charges")
plt.title("Actual vs Predicted Charges for Decision Tree & Random Forest")
plt.legend()
plt.show()

