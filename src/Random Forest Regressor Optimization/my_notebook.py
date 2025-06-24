#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split


# #**Data_Overview**
# 
# 

# In[4]:


data = pd.read_csv("winequalityN.csv")
data.describe()


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# # **Data Preproseccing**

# Handeling non-numeric values
# 

# In[ ]:


x_full= pd.read_csv("/content/drive/MyDrive/MachineLearningProject/winequalityN.csv",index_col="id")
x_full_encoded = pd.get_dummies(x_full,columns =['type'],prefix=['type'])


# In[ ]:


x_full_encoded.head()


# Handeling missing values

# In[ ]:


missing_values = x_full_encoded.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_x= pd.DataFrame(my_imputer.fit_transform(x_full_encoded))
# Imputation removed column names; put them back
imputed_x.columns = x_full_encoded.columns


# In[ ]:


missing_values = imputed_x.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])


# In[ ]:


x_final=imputed_x


# Scaling Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_need_scaling = [x for x in x_final.columns if x !="quality"]
x_final[columns_need_scaling] = scaler.fit_transform(x_final[columns_need_scaling])


# In[ ]:


x_final.head()


# Plots

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for each numeric features
x_final.hist(figsize=(10,10))
plt.show()


# Pie Chart
x_final['quality'].value_counts().plot(kind='pie')


sns.violinplot(x='type_white', y='quality', data=x_final)
plt.show()


# Oversampling

# In[ ]:


x_final["quality"].value_counts()


# In[ ]:


import pandas as pd

# Filter unique values of the target variable 'quality'
valid_classes = [x for x in x_final["quality"].unique() if x in [6, 5, 7]]

# Create DataFrame 'df_whole' containing only the valid classes
df_whole = x_final[x_final['quality'].isin(valid_classes)]

df_whole.quality.unique()


# In[ ]:


count_class_9, count_class_3, count_class_4, count_class_8 = 5, 30, 216, 193
count_class_7 = 1079

valid_classes = [x for x in x_final["quality"].unique() if x in [6, 5, 7]]
df_whole = x_final[x_final['quality'].isin(valid_classes)]

df_class_9 = x_final[x_final["quality"] == 9]
df_class_3 = x_final[x_final["quality"] == 3]
df_class_4 = x_final[x_final["quality"] == 4]
df_class_8 = x_final[x_final["quality"] == 8]
df_class=[df_class_9,df_class_3,df_class_4,df_class_8]

for df in df_class:
  df_over_n = df.sample(count_class_7, replace = True)
  df_whole = pd.concat([df_whole, df_over_n], axis = 0)


# In[ ]:


df_whole.quality.value_counts()
x_final=df_whole


# In[ ]:


x_final['quality'].value_counts().plot(kind='pie')


# # Train & Test

# In[ ]:


x=x_final.drop("quality",axis=1)
y=x_final["quality"]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=11)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree_model =DecisionTreeClassifier()
tree_model.fit(x_train,y_train)
print(tree_model.score(x_valid, y_valid))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_model =DecisionTreeRegressor()
tree_model.fit(x_train,y_train)
print(tree_model.score(x_valid, y_valid))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(n_estimators=1000,max_features=1)
forest_model.fit(x_train, y_train)

print(forest_model.score(x_valid, y_valid))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=1000,max_features=1)
forest_model.fit(x_train, y_train)

print(forest_model.score(x_valid, y_valid))


# # Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

grid={'n_estimators': [int(x) for x in range(100,1000,100)],
      'max_depth':[int(x) for x in range(10,20,1)],
      'max_features': [int(x) for x in range(1,15)]
      }
# Initialize GridSearchCV
MS = GridSearchCV(estimator=RandomForestClassifier(),
                  param_grid=grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1,
                  verbose=2)

# Fit the GridSearchCV to the data
H = MS.fit(x, y)

# Get the best parameters from the GridSearchCV
best_params = H.best_params_


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

grid={'n_estimators': [int(x) for x in range(100,1000,100)],
      'max_depth':[int(x) for x in range(15,20,1)],
      'max_features': [1,'sqrt']
      }
# Initialize GridSearchCV
MS = GridSearchCV(estimator=RandomForestRegressor(),
                  param_grid=grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1,
                  verbose=2)

# Fit the GridSearchCV to the data
H = MS.fit(x, y)

# Get the best parameters from the GridSearchCV
best_params = H.best_params_


# In[ ]:


print(best_params)
M = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                           max_depth=best_params['max_depth'],
                           max_features=best_params['max_features'],
                           )

# Fit the RandomForestClassifier to the data
M.fit(x, y)


# In[ ]:


print(best_params)
M = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                           max_depth=best_params['max_depth'],
                           max_features=best_params['max_features'],
                           )

# Fit the RandomForestClassifier to the data
M.fit(x, y)


# In[ ]:


M.score(x_train, y_train)


# In[ ]:


M.score(x_valid,y_valid)

