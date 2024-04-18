import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(os.path.join(BASE_DIR,'main/data.csv'))

df.columns = df.columns.str.lower().str.replace(' ','_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns :
    df[col] = df[col].str.lower().str.replace(' ','_')
    
df.rename(columns = {'msrp':'price'} ,inplace =True)

pd.options.display.float_format = '{:,.2f}'.format

df['log_price'] = np.log1p(df.price)

np.random.seed(2)

n= len(df)
n_val  =  int(0.2 * n)
n_test =  int(0.2 * n)
n_train = n - (n_val + n_test)


idx = np.arange(n)
np.random.shuffle(idx)
df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

y_train = df_train.log_price.values
y_val = df_val.log_price.values
y_test = df_test.log_price.values

base = ['engine_hp','engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity'] # we take the numerical only here 


# print(df['model'].nunique())
# print(df['model'].value_counts().head(10))

def prepare_X(df):
    df = df.copy()
    features = base.copy()
    
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns :
        df[col] = df[col].str.lower().str.replace(' ','_')
    
    df['age']= 2024 - df.year
    features.append('age')
    
    for v in [2,3,4] :
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
        
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge','nissan','gmc','honda','mazda','cadillac']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
        
    for v in ['regular_unleaded', 'premium_unleaded_(required)', 
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)','diesel','electric']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature) 
    
    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)
        
    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheels_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback','performance','crossover,luxury','luxury,high-performance','exotic,high-performance','hatchback,performance']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback','crew_cab_pickup','extended_cab_pickup','wagon','2dr_hatchback','passenger_minivan']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)
    
    for v in ['silverado_1500','tundra','f-150','sierra_1500','beetle_convertible','corvette','tacoma','gti','frontier','beetle','accord']:
        feature = 'is_model_%s' % v
        df[feature] = (df['model'] == v).astype(int)
        features.append(feature)
    
    
    df_num = df[features]
    df_num = df_num.fillna(df_num.mean())
    X = df_num.values
    return X


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]


# to calculate the error .
def rmse(y,y_pred):
    error = y_pred - y
    mse = (error **2).mean()
    return np.sqrt(mse)


def w_calc ():
    X_train = prepare_X(df_train)
    w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
    return w_0 , w




#### Testing :
# X_train = prepare_X(df_train)
# w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

# y_pred = w_0 + X_train.dot(w)
# print('train', rmse(y_train, y_pred))

# X_val = prepare_X(df_val)
# y_pred = w_0 + X_val.dot(w)
# print('val', rmse(y_val, y_pred))


####  Finaly ... using the model :

# this is a testing data it is in the form:
#{'city_mpg': 18,
#  'driven_wheels': 'all_wheel_drive',
#  'engine_cylinders': 6.0,
#  'engine_fuel_type': 'regular_unleaded',
#  'engine_hp': 268.0,
#  'highway_mpg': 25,
#  'log_price': 10.345638111452145,
#  'make': 'toyota',
#  'market_category': 'crossover,performance',
#  'model': 'venza',
#  'number_of_doors': 4.0,
#  'popularity': 2031,
#  'price': 31120,
#  'transmission_type': 'automatic',
#  'vehicle_size': 'midsize',
#  'vehicle_style': 'wagon',
#  'year': 2013}

# ad = df_test.iloc[8].to_dict()

# X_train = prepare_X(df_train)
# w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

# X_test = prepare_X(pd.DataFrame([ad]))
# y_pred = w_0 + X_test.dot(w)
# suggestion = np.expm1(y_pred)[0].astype(int)

# print("Price is : " , ad['price'])
# print("The prediction is : " , suggestion ,'$')


