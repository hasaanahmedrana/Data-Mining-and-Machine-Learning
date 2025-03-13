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

# %%
data = pd.read_csv("car_price_dataset.csv")

# %%
data.shape

# %%
data.head()

# %%
data.info()

# %%
print("\nMissing values in the dataset:")
data.isnull().sum()

# %%
categoricals = data.select_dtypes(include=['object']).columns.tolist()
categoricals

# %%
for each in categoricals:
    print(each, len(data[each].unique().tolist()))

# %%
threshold = 25
data[categoricals] = data[categoricals].apply(
    lambda each: each.where(each.isin(each.value_counts().nlargest(threshold).index), "Other"))

# %%
for each in categoricals:
    print(each, len(data[each].unique().tolist()))

# %%
data.head()

# %%
data = pd.get_dummies(data, columns=categoricals, drop_first=True)

# %%
X = data.drop(columns=['Price']) # All columns except Target column
y = data['Price'] # Target column

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# %%
Results = {}

# %%
cols =  data.columns.tolist()
cols

# %%
regressor =  LinearRegression()

# %%
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Results['All Variables'] = [mse, r2]
mse, r2

# %%
X_train

# %%
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
cols.remove('Price')
cols.insert(0, 'intercept')

# %%
X_train_const

# %%
model_all = sm.OLS(y_train, X_train_const).fit()
y_pred_all = model_all.predict(X_test_const)

# %%
mse_all = mean_squared_error(y_test, y_pred_all)
r2_all = r2_score(y_test, y_pred_all)
model_all.summary(), mse_all, r2_all

# %%
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

X_opt = X_train_const.copy()

while True:
    model = sm.OLS(y_train, X_opt).fit()
    p_values = model.pvalues

    max_p_value = p_values.max()
    if max_p_value > 0.05:  
        feature_to_remove = p_values.idxmax()
        X_opt.drop(columns=[feature_to_remove], inplace=True)
    else:
        break  


# %%
selected_indices =  X_opt.columns.tolist()
selected_indices = selected_indices[1:]
selected_columns = [cols[each] for each in selected_indices]
print('Selected Columns from backward elimination:', selected_columns)
print(len(selected_columns))

# %%
model_backward = sm.OLS(y_train, X_opt).fit()
y_pred_backward = model_backward.predict(X_test_const[X_opt.columns]) 

mse = mean_squared_error(y_test, y_pred_backward)
r2  = r2_score(y_test, y_pred_backward)
Results['Backward Elimination'] = [mse, r2] 

# %%
print(model_backward.summary())

# %%
selected_features = ['const']
remaining_features = list(X_train.columns)
best_score = 0
while remaining_features:
    scores = {}
    for feature in remaining_features:
        temp_features = selected_features + [feature]
        model = sm.OLS(y_train, X_train_const[temp_features]).fit()
        scores[feature] = model.rsquared
    best_feature = max(scores, key=scores.get)
    if scores[best_feature] > best_score:
        best_score = scores[best_feature]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    else:
        break

X_train_forward = X_train_const[selected_features]
X_test_forward = X_test_const[selected_features]


# %%
model_forward = sm.OLS(y_train, X_train_forward).fit()
y_pred_forward = model_forward.predict(X_test_forward)

mse_forward = mean_squared_error(y_test, y_pred_forward)
r2_forward = r2_score(y_test, y_pred_forward)

# %%
Results['Forward Selection'] = [mse_forward, r2_forward]

# %%
selected_features  = selected_features[1:]
Selected_Columns = [cols[int(each)] for each in selected_features]
print('Selected Columns from forward selection:', Selected_Columns)
print(len(Selected_Columns))

# %%
model_forward.summary()

# %%
def bidirectional_elimination(X=X_train_const, y=y_train, significance_level=0.05):
    selected_features = []
    remaining_features = list(X.columns)
    remaining_features.remove('const')  # Exclude intercept from selection

    best_score = 0  # Track the best model score

    while remaining_features:
        # Forward Selection: Add the most significant feature
        best_feature = None
        best_pval = float('inf')
        
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_temp = sm.add_constant(X[temp_features])
            model = sm.OLS(y, X_temp).fit()
            pval = model.pvalues[feature]

            if pval < best_pval:
                best_pval = pval
                best_feature = feature
                
        if best_pval < significance_level:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            # Backward Elimination: Remove insignificant features while ensuring score improvement
            while len(selected_features) > 0:
                X_temp = sm.add_constant(X[selected_features])
                model = sm.OLS(y, X_temp).fit()
                pvalues = model.pvalues.drop('const')
                
                worst_pval_feature = pvalues.idxmax()
                worst_pval = pvalues.max()
                
                if worst_pval > significance_level and model.rsquared_adj > best_score: #Check if removal improves score
                    selected_features.remove(worst_pval_feature)
                    best_score = model.rsquared_adj  # Update best score
                else:
                    break
        else:
            break

    return selected_features

# %%
selected_indices = bidirectional_elimination()
selected_columns = [cols[int(each)] for each in selected_indices]
print("Selected Features:", selected_columns)
print(len(selected_columns))

# %%
X_train_bidirectional = X_train_const[selected_indices]
X_test_bidirectional = X_test_const[selected_indices]

# %%
model_bidirectional = sm.OLS(y_train, X_train_bidirectional).fit()
y_pred_bidirectional = model_bidirectional.predict(X_test_bidirectional)


# %%
mse_bidirectional = mean_squared_error(y_test, y_pred_bidirectional)
r2_bidirectional = r2_score(y_test, y_pred_bidirectional)
Results['Bidirectional Elimination'] = [mse_bidirectional, r2_bidirectional]

# %%
model_bidirectional.summary()

# %%
for i,j in Results.items():
    print(f'{i}: MSE: {j[0]} R2: {j[1]}')

# %%


