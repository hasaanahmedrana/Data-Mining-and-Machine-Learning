import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import ConfusionMatrixDisplay
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import RocCurveDisplay
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
# Dataset: https://www.kaggle.com/datasets/andrewmvd/divorce-prediction/data
# %%
data = pd.read_csv(r"C:\Users\PMLS\Desktop\SEM-6\Data Mining And Machine Learning\Data-Mining-and-Machine-Learning\Assignment-6\divorce_data.csv", delimiter=";")

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

#%%
X = data.drop(columns=['Divorce'])  # Features
y = data['Divorce']  # Target (Binary Classification)


#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

mi_scores = mutual_info_classif(X_train_scaled, y_train)  # Use only training data
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

top_features = mi_series.head(30).index  
X_selected = X[top_features]

print("Top Features from Mutual Information:\n", top_features)

#%%

X_const = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X_const).fit()  # Fit model

while True:
    p_values = model.pvalues[1:]  # Exclude intercept
    max_p = p_values.max()  # Find max p-value
    if max_p > 0.05:  # Remove feature with highest p-value if > 0.05
        worst_feature = p_values.idxmax()
        X_const.drop(columns=[worst_feature], inplace=True)
        model = sm.OLS(y, X_const).fit()
    else:
        break

selected_features_backward = X_const.columns[1:]  # Exclude intercept
print("\nFinal Selected Features (Backward Elimination):\n", selected_features_backward)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#%%
# Preprocessing: Scale numerical features 
numericals.remove('Divorce')
scaler = StandardScaler()
X_train[numericals] = scaler.fit_transform(X_train[numericals])
X_test[numericals]  = scaler.transform(X_test[numericals])

#%%
# Logistic Regression Model
logreg_model = LogisticRegression(max_iter=2000, random_state=32)
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)

#%%
# Evaluate Logistic Regression Model
def evaluate_model(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")

#%%
evaluate_model("Logistic Regression", y_test, logreg_pred)

#%%
# Plot Confusion Matrix

ConfusionMatrixDisplay.from_predictions(y_test, logreg_pred, cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

#%%
# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, logreg_pred)
plt.title("ROC Curve for Logistic Regression")
plt.show()