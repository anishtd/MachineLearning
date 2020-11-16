# -*- coding: utf-8 -*-
"""

@author: anish
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

dataset.head()

dataset.info() #info of datatype

dataset.boxplot('YearBuilt', rot = 30, figsize=(5,6)) #outlier using boxplot

print(dataset.isnull().values.sum()) # null value total

print(dataset.isnull().sum()) # null value per columns

print(dataset['LotFrontage'].value_counts()) # category type with deceding order
print(dataset['PoolQC'].value_counts().index[1]) # category type highest no
# bar chart
import seaborn as sns
Neighborhood_count = dataset['Neighborhood'].value_counts()
sns.set(style="darkgrid")
sns.barplot(Neighborhood_count.index, Neighborhood_count.values, alpha=1)
plt.title('Frequency Distribution of Neighborhood')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Neighborhood', fontsize=12)
plt.xticks(rotation=90)
plt.show()
# pie chart
labels = dataset['Neighborhood'].astype('category').cat.categories.tolist()
counts = dataset['Neighborhood'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()

# encode values manual
replace_map_compreplace_map = {'Street': {'Pave': 1, 'Grvl': 2}}

# encode with loop
labels = dataset['Neighborhood'].astype('category').cat.categories.tolist()
replace_map_comp = {'Neighborhood' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print(replace_map_comp)


dataset_replace = dataset.copy()

# repalce the column
dataset_replace.replace(replace_map_comp, inplace=True)

print(dataset_replace.head())

# chnage datatype to category
print(dataset_replace['Neighborhood'].dtypes)
dataset_replace['Neighborhood'] = dataset_replace['Neighborhood'].astype('category')
print(dataset_replace['Neighborhood'].dtypes)

dataset_replace.info()
dataset_replace['SaleType'].value_counts()
dataset_replace['LandContour'] = dataset_replace['LandContour'].astype('category')


# encode values auto
dataset_replace['LandContour'] = dataset_replace['LandContour'].cat.codes

dataset_replace['LandSlope'].value_counts()
dataset_replace['LandSlopeContour'] = dataset_replace['LandContour'].cat.codes


# encode specific values
dataset_replace['LandSlope'] = np.where(dataset_replace['LandSlope'].str.contains('Sev'), 3, dataset_replace['LandSlope'])


# encode specific LabelEncoder same as auto
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
dataset_replace['MSZoning'] = lc.fit_transform(dataset_replace['MSZoning'])
# use knn
import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

# start the KNN training
imputed_training=fast_knn(train.values, k=30)

# use Multiple Imputations (MIs) 
from impyute.imputation.cs import mice

# start the MICE training
imputed_training=mice(train.values)


# Deep Neural Networks
import datawig

df_train, df_test = datawig.utils.random_split(train)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['1','2','3','4','5','6','7', 'target'], # column(s) containing information about the column we want to impute
    output_column= '0', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)


# use cluseting

#------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

print(dataset.isnull().sum())

dataset.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1, inplace = True)

dataset.info()
print(dataset['MasVnrType'].value_counts())

dataset_replace = dataset.copy()

sns.heatmap(dataset_replace.isnull(), yticklabels=False,cbar=False,cmap='coolwarm')

dataset_replace['MasVnrType'].fillna(dataset_replace['MasVnrType'].mode()[0])


from sklearn.preprocessing import LabelEncoder 
lc = LabelEncoder()
dataset_replace['MasVnrType'] = lc.fit_transform(dataset_replace['MasVnrType'])

replace_map = {'MasVnrType': {'None': 0, 'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['MasVnrType'] = dataset_replace['MasVnrType'].astype('float64')

print(dataset['BsmtQual'].value_counts())
dataset_replace['BsmtQual'] = dataset_replace['BsmtQual'].astype('str')
replace_map = {'BsmtQual': {'TA': 1, 'Gd': 2, 'Ex': 3, 'Fa': 4}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['BsmtQual'] = dataset_replace['BsmtQual'].astype('float64')

print(dataset['BsmtCond'].value_counts())
dataset_replace['BsmtCond'] = dataset_replace['BsmtCond'].astype('str')
replace_map = {'BsmtCond': {'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['BsmtCond'] = dataset_replace['BsmtCond'].astype('float64')

print(dataset['BsmtExposure'].value_counts())
dataset_replace['BsmtExposure'] = dataset_replace['BsmtExposure'].astype('str')
replace_map = {'BsmtExposure': {'No': 1, 'Av': 2, 'Gd': 3, 'Mn': 4}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['BsmtExposure'] = dataset_replace['BsmtExposure'].astype('float64')

print(dataset['BsmtFinType1'].value_counts())
dataset_replace['BsmtFinType1'] = dataset_replace['BsmtFinType1'].astype('str')
replace_map = {'BsmtFinType1': {'Unf': 1, 'GLQ': 2, 'ALQ': 3, 'BLQ': 4, 'Rec': 5, 'LwQ': 6}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['BsmtFinType1'] = dataset_replace['BsmtFinType1'].astype('float64')

print(dataset['BsmtFinType2'].value_counts())
dataset_replace['BsmtFinType2'] = dataset_replace['BsmtFinType2'].astype('str')
replace_map = {'BsmtFinType2': {'Unf': 1, 'GLQ': 2, 'ALQ': 3, 'BLQ': 4, 'Rec': 5, 'LwQ': 6}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['BsmtFinType2'] = dataset_replace['BsmtFinType2'].astype('float64')

print(dataset['Electrical'].value_counts())
dataset_replace['Electrical'] = dataset_replace['Electrical'].astype('str')
replace_map = {'Electrical': {'SBrkr': 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['Electrical'] = dataset_replace['Electrical'].astype('float64')

print(dataset['GarageType'].value_counts())
dataset_replace['GarageType'] = dataset_replace['GarageType'].astype('str')
replace_map = {'GarageType': {'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3, 'Basment': 4, 'CarPort': 5, '2Types': 6}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['GarageType'] = dataset_replace['GarageType'].astype('float64')

print(dataset['GarageFinish'].value_counts())
dataset_replace['GarageFinish'] = dataset_replace['GarageFinish'].astype('str')
replace_map = {'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['GarageFinish'] = dataset_replace['GarageFinish'].astype('float64')

print(dataset['GarageQual'].value_counts())
dataset_replace['GarageQual'] = dataset_replace['GarageQual'].astype('str')
replace_map = {'GarageQual': {'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4, 'Ex': 5}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['GarageQual'] = dataset_replace['GarageQual'].astype('float64')

print(dataset['GarageCond'].value_counts())
dataset_replace['GarageCond'] = dataset_replace['GarageCond'].astype('str')
replace_map = {'GarageCond': {'TA': 1, 'Gd': 2, 'Fa': 3, 'Po': 4, 'Ex': 5}}
dataset_replace.replace(replace_map, inplace=True)
dataset_replace['GarageCond'] = dataset_replace['GarageCond'].astype('float')

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

imp3 = KNNImputer(n_neighbors = 3)
dataset_replace["GarageFinish"] = imp3.fit_transform(dataset_replace[["GarageFinish"]])

imp4 = KNNImputer(n_neighbors = 4)
dataset_replace["BsmtQual"] = imp4.fit_transform(dataset_replace[["BsmtQual"]])
dataset_replace["BsmtCond"] = imp4.fit_transform(dataset_replace[["BsmtCond"]])
dataset_replace["MasVnrType"] = imp4.fit_transform(dataset_replace[["MasVnrType"]])
dataset_replace["BsmtExposure"] = imp4.fit_transform(dataset_replace[["BsmtExposure"]])


imp5 = KNNImputer(n_neighbors = 5)
dataset_replace["Electrical"] = imp5.fit_transform(dataset_replace[["Electrical"]])
dataset_replace["GarageQual"] = imp5.fit_transform(dataset_replace[["GarageQual"]])
dataset_replace["GarageCond"] = imp5.fit_transform(dataset_replace[["GarageCond"]])

imp6 = KNNImputer(n_neighbors = 6)
dataset_replace["BsmtFinType1"] = imp6.fit_transform(dataset_replace[["BsmtFinType1"]])
dataset_replace["BsmtFinType2"] = imp6.fit_transform(dataset_replace[["BsmtFinType2"]])
dataset_replace["GarageType"] = imp6.fit_transform(dataset_replace[["GarageType"]])

labels = dataset_replace['MSZoning'].astype('category').cat.categories.tolist()
replace_map_comp = {'MSZoning' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Street'].astype('category').cat.categories.tolist()
replace_map_comp = {'Street' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['LotShape'].astype('category').cat.categories.tolist()
replace_map_comp = {'LotShape' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['LandContour'].astype('category').cat.categories.tolist()
replace_map_comp = {'LandContour' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Utilities'].astype('category').cat.categories.tolist()
replace_map_comp = {'Utilities' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['LotConfig'].astype('category').cat.categories.tolist()
replace_map_comp = {'LotConfig' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['LandSlope'].astype('category').cat.categories.tolist()
replace_map_comp = {'LandSlope' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Neighborhood'].astype('category').cat.categories.tolist()
replace_map_comp = {'Neighborhood' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Condition1'].astype('category').cat.categories.tolist()
replace_map_comp = {'Condition1' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Condition2'].astype('category').cat.categories.tolist()
replace_map_comp = {'Condition2' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['BldgType'].astype('category').cat.categories.tolist()
replace_map_comp = {'BldgType' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['HouseStyle'].astype('category').cat.categories.tolist()
replace_map_comp = {'HouseStyle' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['RoofStyle'].astype('category').cat.categories.tolist()
replace_map_comp = {'RoofStyle' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['RoofMatl'].astype('category').cat.categories.tolist()
replace_map_comp = {'RoofMatl' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Exterior1st'].astype('category').cat.categories.tolist()
replace_map_comp = {'Exterior1st' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Exterior2nd'].astype('category').cat.categories.tolist()
replace_map_comp = {'Exterior2nd' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['MasVnrType'].astype('category').cat.categories.tolist()#..................
#replace_map_comp = {'MasVnrType' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['ExterQual'].astype('category').cat.categories.tolist()
replace_map_comp = {'ExterQual' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['ExterCond'].astype('category').cat.categories.tolist()
replace_map_comp = {'ExterCond' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Foundation'].astype('category').cat.categories.tolist()
replace_map_comp = {'Foundation' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['BsmtQual'].astype('category').cat.categories.tolist()
#replace_map_comp = {'BsmtQual' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True) #....

#labels = dataset_replace['BsmtCond'].astype('category').cat.categories.tolist()
#replace_map_comp = {'BsmtCond' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['BsmtExposure'].astype('category').cat.categories.tolist()
#replace_map_comp = {'BsmtExposure' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['BsmtFinType1'].astype('category').cat.categories.tolist()
#replace_map_comp = {'BsmtFinType1' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['BsmtFinType2'].astype('category').cat.categories.tolist()
#replace_map_comp = {'BsmtFinType2' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Heating'].astype('category').cat.categories.tolist()
replace_map_comp = {'Heating' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['HeatingQC'].astype('category').cat.categories.tolist()
replace_map_comp = {'HeatingQC' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['CentralAir'].astype('category').cat.categories.tolist()
replace_map_comp = {'CentralAir' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['Electrical'].astype('category').cat.categories.tolist()
#replace_map_comp = {'Electrical' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)
#dataset_replace.info()

labels = dataset_replace['KitchenQual'].astype('category').cat.categories.tolist()
replace_map_comp = {'KitchenQual' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['Functional'].astype('category').cat.categories.tolist()
replace_map_comp = {'Functional' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['GarageType'].astype('category').cat.categories.tolist()
#replace_map_comp = {'GarageType' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['GarageFinish'].astype('category').cat.categories.tolist()
#replace_map_comp = {'GarageFinish' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['GarageQual'].astype('category').cat.categories.tolist()
#replace_map_comp = {'GarageQual' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['PavedDrive'].astype('category').cat.categories.tolist()
replace_map_comp = {'PavedDrive' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

#labels = dataset_replace['GarageCond'].astype('category').cat.categories.tolist()
#replace_map_comp = {'GarageCond' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
#odataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['SaleType'].astype('category').cat.categories.tolist()
replace_map_comp = {'SaleType' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

labels = dataset_replace['SaleCondition'].astype('category').cat.categories.tolist()
replace_map_comp = {'SaleCondition' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
dataset_replace.replace(replace_map_comp, inplace=True)

dataset_replace.info()
dataset_replace.columns



correlation_mat = dataset_replace.corr()
sns.heatmap(correlation_mat, annot = True)
plt.rc('axes', grid=True)
plt.rc('figure', figsize=(100, 80))
plt.rc('legend', fancybox=True, framealpha=1)
plt.show()

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

categorical_cols = ['MSSubClass', 'MSZoning', 'LotArea', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC','CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd','Functional', 'Fireplaces', 'GarageType', 'GarageFinish','GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType','SaleCondition']
df.info()
transformed_data = onehotencoder.fit_transform(dataset_replace[categorical_cols])

import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

# start the KNN training
imputed_training=fast_knn(dataset_replace.values, k=30)
df = pd.DataFrame(imputed_training)
df.info()

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
lm2.score(X, y)
print(regressor.intercept_)
print(regressor.coef_)

df_result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_result

df1 = df_result.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

xt = sc_x.fit_transform(x_train)
yt = sc_y.fit_transform(y_train.reshape(len(y_train),1))
     
regressor1 = LinearRegression()
regressor1.fit(xt, y_train)

yt_pred = regressor1.predict(sc_x.transform(x_test))
yt_predi = sc_y.inverse_transform(yt_pred)

print('Mean Absolute Error:', metrics.mean_absolute_error(sc_y.transform(y_test.reshape(len(y_test),1)), yt_predi))  
print('Mean Squared Error:', metrics.mean_squared_error(sc_y.transform(y_test.reshape(len(y_test),1)), yt_predi)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(sc_y.transform(y_test.reshape(len(y_test),1)), yt_predi)))

dft_result = pd.DataFrame({'Actual': sc_y.transform(y_test.reshape(len(y_test),1)).flatten(), 'Predicted': yt_predi.flatten()})
dft_result

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, yt_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, yt_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, yt_pred)))

dft_result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yt_pred.flatten()})
dft_result

df1 = df_result.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print(xt)
import statsmodels.formula.api as sm
model = sm.ols(formula='Y ~ x1+x2+x3', data=adj_sample)
fitted1 = model.fit()
fitted1.summary()
#-------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

dataset.drop(['Id','LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1, inplace = True)

dataset.info()
print(dataset['MasVnrType'].value_counts())

dataset_replace = dataset.copy()

sns.heatmap(dataset_replace.isnull(), yticklabels=False,cbar=False,cmap='coolwarm')

dataset_replace['MasVnrType'] = dataset_replace['MasVnrType'].fillna(dataset_replace['MasVnrType'].mode()[0])
dataset_replace['BsmtQual'] = dataset_replace['BsmtQual'].fillna(dataset_replace['BsmtQual'].mode()[0])
dataset_replace['BsmtCond'] = dataset_replace['BsmtCond'].fillna(dataset_replace['BsmtCond'].mode()[0])
dataset_replace['BsmtExposure'] = dataset_replace['BsmtExposure'].fillna(dataset_replace['BsmtExposure'].mode()[0])
dataset_replace['BsmtFinType1'] = dataset_replace['BsmtFinType1'].fillna(dataset_replace['BsmtFinType1'].mode()[0])
dataset_replace['BsmtFinType2'] = dataset_replace['BsmtFinType2'].fillna(dataset_replace['BsmtFinType2'].mode()[0])
dataset_replace['Electrical'] = dataset_replace['Electrical'].fillna(dataset_replace['Electrical'].mode()[0])
dataset_replace['GarageType'] = dataset_replace['GarageType'].fillna(dataset_replace['GarageType'].mode()[0])
dataset_replace['GarageFinish'] = dataset_replace['GarageFinish'].fillna(dataset_replace['GarageFinish'].mode()[0])
dataset_replace['GarageQual'] = dataset_replace['GarageQual'].fillna(dataset_replace['GarageQual'].mode()[0])
dataset_replace['GarageCond'] = dataset_replace['GarageCond'].fillna(dataset_replace['GarageCond'].mode()[0])

dataset_replace['MasVnrArea'] = dataset_replace['MasVnrArea'].fillna(dataset_replace['MasVnrArea'].mean())
dataset_replace['GarageYrBlt'] = dataset_replace['GarageYrBlt'].fillna(dataset_replace['GarageYrBlt'].mean())

dataset_replace.info()

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

categorical_cols = ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC','CentralAir', 'Electrical', 'KitchenQual','Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType','SaleCondition']
#df.info()
#transformed_data = onehotencoder.fit_transform(dataset_replace[categorical_cols])
#print(transformed_data)
#encoded_data = pd.DataFrame(transformed_data, index=dataset_replace.index)
#concatenated_data = pd.concat([data, encoded_data], axis=1)

#pd.get_dummies(dataset_replace, columns=categorical_cols)
#-------------------------------------------
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
dataset_replace[categorical_cols] = dataset_replace[categorical_cols].apply(lambda col: le.fit_transform(col))    
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

#One-hot-encode the categorical columns.
#Unfortunately outputs an array instead of dataframe.
array_hot_encoded = ohe.fit_transform(dataset_replace[categorical_cols])

#Convert it to df
data_hot_encoded = pd.DataFrame(array_hot_encoded, index=dataset_replace.index)

#Extract only the columns that didnt need to be encoded
data_other_cols = dataset_replace.drop(columns=categorical_cols)

#Concatenate the two dataframes : 
data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)
#------------------------------------------------
df=dataset_replace.copy()

def category_onehot_multicol(multicols):
    df_final = df
    i=0
    for fields in multicols:
        print(fields)
        df1=pd.get_dummies(df[fields], drop_first=True)
        df.drop([fields], axis = 1, inplace=True)
        if i==0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i=i+1
        
    df_final = pd.concat([df,df_final], axis=1)
    return df_final

final_df = category_onehot_multicol(categorical_cols)

final_df = final_df.iloc[:,~final_df.columns.duplicated()]

x = final_df.drop(['SalePrice'], axis=1)
#y = final_df['SalePrice']
y = final_df.iloc[:,35:36].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df_result = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_result

#------------

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

xt = sc_x.fit_transform(x_train)
yt = sc_y.fit_transform(y_train)
     
regressor1 = LinearRegression()
regressor1.fit(xt, yt)

yt_pred = regressor1.predict(sc_x.transform(x_test))
yt_predi = sc_y.inverse_transform(yt_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, yt_predi))
print(rms)

print('Mean Absolute Error:', metrics.mean_absolute_error(sc_y.transform(y_test), yt_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(sc_y.transform(y_test), yt_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(sc_y.transform(y_test), yt_pred)))

dft_result = pd.DataFrame({'Actual': sc_y.transform(y_test.reshape(len(y_test),1)).flatten(), 'Predicted': yt_predi.flatten()})
dft_result

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, yt_predi))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, yt_predi))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, yt_predi)))

dft_result = pd.DataFrame({'Actual': sc_y.transform(y_test).flatten(), 'Predicted': yt_pred.flatten()})
dft_result
#x = final_df.iloc[:,30:35].values
#y = final_df.iloc[:,-35].values
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split

cols = final_df.columns.tolist()
cols = ['MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','FV','RH','RL','RM','Pave','IR2','IR3','Reg','HLS','Low','Lvl','NoSeWa','CulDSac','FR2','FR3','Inside','Mod', 'Sev','Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel','NAmes','NPkVill','NWAmes','NoRidge','NridgHt','OldTown','SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker','Feedr','Norm','PosA','PosN','RRAe','RRAn','RRNe','RRNn','2fmCon','Duplex','Twnhs','TwnhsE','1.5Unf','1Story','2.5Fin','2.5Unf','2Story','SFoyer','SLvl','Gable','Gambrel','Hip','Mansard','Shed','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc','MetalSd','Plywood','Stone','Stucco','VinylSd','Wd Sdng','WdShing','Brk Cmn','CmentBd','Other','Wd Shng','None','Fa','Gd','TA','Po','PConc','Slab','Wood','Mn','No','BLQ','GLQ','LwQ','Rec','Unf','GasA','GasW','Grav','OthW','Wall','Y','FuseF','FuseP','Mix','SBrkr','Maj2','Min1','Min2','Typ','Attchd','Basment','BuiltIn','CarPort','Detchd','RFn','P','CWD','Con','ConLD','ConLI','ConLw','New','Oth','WD','AdjLand','Alloca','Family','Normal','Partial', 'SalePrice',]
df = final_df[cols]

X, X_test, y, y_test = train_test_split(df.iloc[:,:175],
                                        df.iloc[:,-1],
                                        test_size=0.20,
                                        random_state=42)
from sklearn.tree import DecisionTreeRegressor

dtm = DecisionTreeRegressor(max_depth=4, min_samples_split=5, max_leaf_nodes=10)

dtm.fit(X,y)
print("R-Squared on train dataset={}".format(dtm.score(X_test,y_test)))

dtm.fit(X_test,y_test)
print("R-Squaredon test dataset={}".format(dtm.score(X_test,y_test)))


param_grid = {"criterion": ["mse", "mae"],
              "min_samples_split": [4, 8, 12],
              "max_depth": [6,8,12],
              "min_samples_leaf": [10, 20, 40],
              "max_leaf_nodes":[20, 40, 100]
    }
grid_cv = GridSearchCV(dtm, param_grid, cv=10)
grid_cv.fit(X,y)

print("R-Sqaure",format(grid_cv.best_score_))
print("Best Hyperparameters", format(grid_cv.best_params_))

dfr = pd.DataFrame(data=grid_cv.cv_results_)
dfr.head()

r2_scores = cross_val_score(grid_cv.best_estimator_, X, y, cv=10)
mse_scores = cross_val_score(grid_cv.best_estimator_, X, y, cv=10,scoring='neg_mean_squared_error')

print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))

df = pd.DataFrame(data=grid_cv.cv_results_)
df.head()

fig,ax = plt.subplots()
sns.pointplot(data=df[['mean_test_score',
                           'param_max_leaf_nodes',
                           'param_max_depth']],
             y='mean_test_score',x='param_max_depth',
             hue='param_max_leaf_nodes',ax=ax)
ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")

predicted = grid_cv.best_estimator_.predict(X)
residuals = y.flatten()-predicted

fig, ax = plt.subplots()
ax.scatter(y.flatten(), residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()

# Checking the training model scores
r2_scores = cross_val_score(grid_cv.best_estimator_, X, y, cv=10)
mse_scores = cross_val_score(grid_cv.best_estimator_, X, y, cv=10,scoring='neg_mean_squared_error')

print("avg R-squared::{:.3f}".format(np.mean(r2_scores)))
print("MSE::{:.3f}".format(np.mean(mse_scores)))

#Test dataset evaluation
best_dtm_model = grid_cv.best_estimator_

y_pred = best_dtm_model.predict(X_test)
residuals = y_test.flatten() - y_pred   


r2_score = best_dtm_model.score(X_test,y_test)
print("R-squared:{:.3f}".format(r2_score))
print("MSE: %.2f" % metrics.mean_squared_error(y_test, y_pred))

#--------------------random forest-----------------
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=10)

from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [5,10,50,100,200,300,400],
              'max_depth': [5,10,50,100,200],
              'min_samples_leaf': [5,10,20],
              'max_features': ['auto','log2']
               }
rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                       cv=5,
                       n_jobs=1, 
                       verbose=0, 
                       scoring="neg_mean_squared_error", 
                       return_train_score=True)
rf_grid.fit(X,y)
print(rf_grid.best_params_)

# Extract best model from 'rf_grid'
best_model = rf_grid.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error as MSE
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
#----------------------

print(rf_grid.score(X_test, y_test))

# xgboost

train = df.iloc[:1000,:]
test = df.iloc[1000:,:]

x = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']
x_test = test.drop(['SalePrice'],axis=1)
y_test = test['SalePrice']

import xgboost as xgb
classifier = xgb.XGBClassifier()

regressor = xgb.XGBRegressor()
regressor.fit(x,y)
y_pred = regressor.predict(test)

from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
#---------------------------------------------------------------------------    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

df.head()
df['MSZoning'].value_counts()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.shape
df.info()
## Fill Missing Values
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df.shape
df.drop(['Id'],axis=1,inplace=True)
df.isnull().sum()
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)

#test cleaning
test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df.drop(['Alley'],axis=1,inplace=True)
test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])
test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])
test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])
test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])
test_df.drop(['GarageYrBlt'],axis=1,inplace=True)
test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

nulltest = test_df.isnull().sum()
nulltrain = df.isnull().sum()
test_df.drop(['Id'],axis=1,inplace=True)

test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])
test_df.dropna(inplace=True)
test_df['BsmtExposure'].isnull().count()
df.shape
df.head()
##HAndle Categorical Features
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
main_df=df.copy()
## Combine Test Data 


test_df.shape
test_df.head()
final_df=pd.concat([df,test_df],axis=0)
final_df['SalePrice']
final_df.shape
final_df=category_onehot_multcols(columns)
final_df.shape
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape
df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]

df_Test['SalePrice'].isnull().count()
df_Train.head()

df_Train.shape
y_test = pd.read_csv('house-prices-advanced-regression-techniques/sample_submission.csv')
y_test = y_test['SalePrice']
df_Test.drop(['SalePrice'],axis=1,inplace=True)

X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
df_Test.shape
#--------------------------------------------------------

import xgboost
regressor=xgboost.XGBRegressor()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(df_Test)

from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

from sklearn.metrics import accuracy_score
regressor.score(t, t1)
np.reshape(y_test, y_test.len,1)

t = np.array(y_test.values.tolist()).reshape(len(y_pred),1)
t1 = y_pred.reshape(len(y_pred),1)
y_pred.reshape(len(y_pred),1)

n_estimators = [100,500,900,1500,1100]
max_depth = [2,3,5,10,15]
#---------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ReLU,LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units=50, kernel_initializer='he_uniform', activation='relu', input_dim=174))

classifier.add(Dense(units=25, kernel_initializer='he_uniform', activation='relu'))

classifier.add(Dense(units=50, kernel_initializer='he_uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='he_uniform'))

classifier.compile(loss=root_mean_square_error, optimizer='Adamax')

model_history = classifier.fit(X_train.values, y_train.values, validation_split=0.20,batch_size=10,epochs=1000)

from keras import backend as K
def root_mean_square_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

ann_pred=classifier.predict(df_Test.values)

from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, ann_pred)**(1/2)

history = model_history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['loss'][-1], loss=history['val_loss'][-1]))
