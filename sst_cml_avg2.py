import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner, ForestDRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import expon
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from dowhy.causal_estimator import CausalEstimate
from sklearn.preprocessing import StandardScaler
from econml.dr import DRLearner
from sklearn.linear_model import LassoCV
from econml.dml import DML, SparseLinearDML
import matplotlib.pyplot as plt
import pandas as pd



# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')

#%%

# Set display options for pandas
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create DataFrame for ATE results
data_ATE = pd.DataFrame(0.0, index=range(0, 4), columns=['ATE', 'ci_lower', 'ci_upper']).astype({'ATE': 'float64'})
print(data_ATE)


#%%

# Import data
data = pd.read_csv("D:/data_final_13_23.csv", encoding='latin-1')

data = data[['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI', 'cpolr', 'wpac850', 'cpac850', 'epac850', 'qbo_u30',
             't2m', 'tp', 'MPI', 'pop_density', 'excess', 'Year', 'Month', 'altitude', 'consensus',
             'DANE', 'DANE_year', 'DANE_period', 'Period', 'cases']]

data = data[data['Year'] >= 2013]


#%%

# 1. Label Encoding DANE
le = LabelEncoder()
data['DANE_labeled'] = le.fit_transform(data['DANE'])
scaler = MinMaxScaler()
data['DANE_normalized'] = scaler.fit_transform(
    data[['DANE_labeled']]
)

# 2. Label Encoding DANE_year
le_year = LabelEncoder()
data['DANE_year_labeled'] = le_year.fit_transform(data['DANE_year'])
scaler_year = MinMaxScaler()
data['DANE_year_normalized'] = scaler_year.fit_transform(
    data[['DANE_year_labeled']]
)

# 3. Label Encoding DANE_year_month
le_period = LabelEncoder()
data['DANE_period_labeled'] = le_period.fit_transform(data['DANE_period'])
scaler_period = MinMaxScaler()
data['DANE_period_normalized'] = scaler_period.fit_transform(
    data[['DANE_period_labeled']]
)

#%%

# transform year and month
data.Year = data.Year - 2013
data["sin_month"] = np.sin(2 * np.pi * data["Month"] / 12)
data["cos_month"] = np.cos(2 * np.pi * data["Month"] / 12)

#%%

# SST in t+1
data['SST12_t1'] = data.groupby('DANE')['SST12'].shift(-1)
data['SST3_t1'] = data.groupby('DANE')['SST3'].shift(-1)
data['SST34_t1'] = data.groupby('DANE')['SST34'].shift(-1)
data['SST4_t1'] = data.groupby('DANE')['SST4'].shift(-1)

#%%

# moving average variables       
variables = ['SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'SOI', 'ESOI', 'cpolr', 'wpac850', 'cpac850', 'epac850', 'qbo_u30', 't2m', 'tp']

# Tamaños de window deseados
windows = [2]

for var in variables:
    for window in windows:
        # new column
        nueva_col = f'{var}_avg{window}'
        
        # moving average
        data[nueva_col] = data.groupby('DANE')[var].transform(
            lambda x: x.rolling(window=window, min_periods=1, closed='right').mean()
        )

print(data.columns)
data.head(10)

scaler = StandardScaler()
data['SST12_avg2'] = scaler.fit_transform(data[['SST12_avg2']])
data['SST12_t1'] = scaler.fit_transform(data[['SST12_t1']])
data['SST3_avg2'] = scaler.fit_transform(data[['SST3_avg2']])
data['SST3_t1'] = scaler.fit_transform(data[['SST3_t1']])
data['SST34_avg2'] = scaler.fit_transform(data[['SST34_avg2']])
data['SST34_t1'] = scaler.fit_transform(data[['SST34_t1']])
data['SST4_avg2'] = scaler.fit_transform(data[['SST4_avg2']])
data['SST4_t1'] = scaler.fit_transform(data[['SST4_t1']])
data['NATL_avg2'] = scaler.fit_transform(data[['NATL_avg2']])
data['t2m_avg2'] = scaler.fit_transform(data[['t2m_avg2']])
data['tp_avg2'] = scaler.fit_transform(data[['tp_avg2']])
data['MPI'] = scaler.fit_transform(data[['MPI']])
data['pop_density'] = scaler.fit_transform(data[['pop_density']])


# Convert columns to binary
columns_convert = ['SATL_avg2', 'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'TROP_avg2', 'NATL_avg2'] 
for col in columns_convert:
    median = data[col].median()
    data[col] = (data[col] > median).astype(int)
    
data.head(10)

#%%

# Histogram
# For DANE = 5001
data_unique = data[data['DANE'] == 5001]

columnas = ['SST12', 'SST3', 'SST34', 'SST4']
titulos = ['a', 'b', 'c', 'd']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  

for i, (col, titulo) in enumerate(zip(columnas, titulos)):
    axes[i].hist(data_unique[col], bins=50, edgecolor='black', alpha=0.7) 
    axes[i].set_title(titulo, fontsize=14)
    axes[i].set_xlabel('Temperature °C')
    axes[i].set_ylabel('Frecuency')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

for col in columnas:
    sd = data_unique[col].std()
    print(f"Standard deviation of {col}: {sd:.4f}")
    
#%%

# Time series
total_cases_by_period = data.groupby('Period')['cases'].sum().reset_index()
total_cases_by_period['Year'] = 2013 + (total_cases_by_period['Period'] - 1) // 12

plt.style.use('seaborn-v0_8-darkgrid') 
fig, ax = plt.subplots(figsize=(15, 8)) 

ax.plot(total_cases_by_period['Period'], total_cases_by_period['cases'], 
        marker='o', 
        linestyle='-', 
        color='#1f77b4',
        linewidth=1.5, 
        markersize=4, 
        markerfacecolor='lightblue', 
        markeredgecolor='darkblue', 
        markeredgewidth=0.5)

unique_years = total_cases_by_period['Year'].unique()
year_positions = []
year_labels = []

for year in unique_years:
    first_period_of_year = (year - 2013) * 12 + 1
    if first_period_of_year <= 132:  
        year_positions.append(first_period_of_year)
        year_labels.append(str(year))

ax.set_xticks(year_positions)
ax.set_xticklabels(year_labels)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Cases', fontsize=12)


ax.legend(loc='upper left') 


ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)


plt.tight_layout()
plt.show()

  
#%%        

data_SST12_avg2 = data[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST12_avg2', 'SST3_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST12_t1']]

data_SST12_avg2 = data_SST12_avg2.dropna()

# Other SST as binary
columns_convert = ['SST3_avg2', 'SST34_avg2', 'SST4_avg2'] 
for col in columns_convert:
    median = data_SST12_avg2[col].median()
    data_SST12_avg2[col] = (data_SST12_avg2[col] > median).astype(int)


data_SST12_avg2.head(10)

#%%

#Causal mechanism
model_SST12 = CausalModel(
        data = data_SST12_avg2,
        treatment=['SST12_avg2'],
        outcome=['excess'],
        effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST12_t1'],
        common_causes=['SST3_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                       'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2',
                       'Year', 'sin_month', 'cos_month'], 
        graph= """graph[directed 1 
                    node[id "SST12_avg2" label "SST12_avg2"]
                    node[id "excess" label "excess"]
                    node[id "SST3_avg2" label "SST3_avg2"]
                    node[id "SST34_avg2" label "SST34_avg2"]
                    node[id "SST4_avg2" label "SST4_avg2"]
                    node[id "NATL_avg2" label "NATL_avg2"]
                    node[id "SATL_avg2" label "SATL_avg2"]
                    node[id "TROP_avg2" label "TROP_avg2"]
                    node[id "SOI_avg2" label "SOI_avg2"]
                    node[id "ESOI_avg2" label "ESOI_avg2"]
                    node[id "cpolr_avg2" label "cpolr_avg2"]
                    node[id "wpac850_avg2" label "wpac850_avg2"]
                    node[id "cpac850_avg2" label "cpac850_avg2"]
                    node[id "epac850_avg2" label "epac850_avg2"]
                    node[id "qbo_u30_avg2" label "qbo_u30_avg2"]
                    node[id "t2m_avg2" label "t2m_avg2"]
                    node[id "tp_avg2" label "tp_avg2"]
                    node[id "MPI" label "MPI"]
                    node[id "pop_density" label "pop_density"]
                    node[id "Year" label "Year"]
                    node[id "sin_month" label "sin_month"]
                    node[id "cos_month" label "cos_month"]
                    node[id "DANE_normalized" label "DANE_normalized"]
                    node[id "DANE_year_normalized" label "DANE_year_normalized"]
                    node[id "DANE_period_normalized" label "DANE_period_normalized"]
                    node[id "altitude" label "altitude"]
                                       
                    edge[source "Year" target "SST12_avg2"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3_avg2"]
                    edge[source "Year" target "SST34_avg2"]
                    edge[source "Year" target "SST4_avg2"]
                    edge[source "Year" target "NATL_avg2"]
                    edge[source "Year" target "SATL_avg2"]
                    edge[source "Year" target "TROP_avg2"]
                    edge[source "Year" target "SOI_avg2"]
                    edge[source "Year" target "ESOI_avg2"]
                    edge[source "Year" target "cpolr_avg2"]
                    edge[source "Year" target "wpac850_avg2"]
                    edge[source "Year" target "cpac850_avg2"]
                    edge[source "Year" target "epac850_avg2"]
                    edge[source "Year" target "qbo_u30_avg2"]
                    edge[source "Year" target "t2m_avg2"]
                    edge[source "Year" target "tp_avg2"]
                    
        		    edge[source "sin_month" target "SST12_avg2"]
                    edge[source "sin_month" target "excess"]
                    edge[source "sin_month" target "SST3_avg2"]
                    edge[source "sin_month" target "SST34_avg2"]
                    edge[source "sin_month" target "SST4_avg2"]
                    edge[source "sin_month" target "NATL_avg2"]
                    edge[source "sin_month" target "SATL_avg2"]
                    edge[source "sin_month" target "TROP_avg2"]
                    edge[source "sin_month" target "SOI_avg2"]
                    edge[source "sin_month" target "ESOI_avg2"]
                    edge[source "sin_month" target "cpolr_avg2"]
                    edge[source "sin_month" target "wpac850_avg2"]
                    edge[source "sin_month" target "cpac850_avg2"]
                    edge[source "sin_month" target "epac850_avg2"]
                    edge[source "sin_month" target "qbo_u30_avg2"]
                    edge[source "sin_month" target "t2m_avg2"]
                    edge[source "sin_month" target "tp_avg2"]

                    edge[source "cos_month" target "SST12_avg2"]
                    edge[source "cos_month" target "excess"]
                    edge[source "cos_month" target "SST3_avg2"]
                    edge[source "cos_month" target "SST34_avg2"]
                    edge[source "cos_month" target "SST4_avg2"]
                    edge[source "cos_month" target "NATL_avg2"]
                    edge[source "cos_month" target "SATL_avg2"]
                    edge[source "cos_month" target "TROP_avg2"]
                    edge[source "cos_month" target "SOI_avg2"]
                    edge[source "cos_month" target "ESOI_avg2"]
                    edge[source "cos_month" target "cpolr_avg2"]
                    edge[source "cos_month" target "wpac850_avg2"]
                    edge[source "cos_month" target "cpac850_avg2"]
                    edge[source "cos_month" target "epac850_avg2"]
                    edge[source "cos_month" target "qbo_u30_avg2"]
                    edge[source "cos_month" target "t2m_avg2"]
                    edge[source "cos_month" target "tp_avg2"]
                    
                           
                    edge[source "SST3_avg2" target "SST34_avg2"]
                    edge[source "SST3_avg2" target "SST4_avg2"]
                    edge[source "SST3_avg2" target "NATL_avg2"]
                    edge[source "SST3_avg2" target "SATL_avg2"]
                    edge[source "SST3_avg2" target "TROP_avg2"]
                    edge[source "SST3_avg2" target "SOI_avg2"]
                    edge[source "SST3_avg2" target "ESOI_avg2"]
                    edge[source "SST3_avg2" target "cpolr_avg2"]
                    edge[source "SST3_avg2" target "wpac850_avg2"]
                    edge[source "SST3_avg2" target "cpac850_avg2"]
                    edge[source "SST3_avg2" target "epac850_avg2"]
                    edge[source "SST3_avg2" target "qbo_u30_avg2"]                    
                    
                    edge[source "SST34_avg2" target "SST4_avg2"]
                    edge[source "SST34_avg2" target "NATL_avg2"]
                    edge[source "SST34_avg2" target "SATL_avg2"]
                    edge[source "SST34_avg2" target "TROP_avg2"]
                    edge[source "SST34_avg2" target "SOI_avg2"]
                    edge[source "SST34_avg2" target "ESOI_avg2"]
                    edge[source "SST34_avg2" target "cpolr_avg2"]
                    edge[source "SST34_avg2" target "wpac850_avg2"]
                    edge[source "SST34_avg2" target "cpac850_avg2"]
                    edge[source "SST34_avg2" target "epac850_avg2"]
                    edge[source "SST34_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST4_avg2" target "NATL_avg2"]
                    edge[source "SST4_avg2" target "SATL_avg2"]
                    edge[source "SST4_avg2" target "TROP_avg2"]
                    edge[source "SST4_avg2" target "SOI_avg2"]
                    edge[source "SST4_avg2" target "ESOI_avg2"]
                    edge[source "SST4_avg2" target "cpolr_avg2"]
                    edge[source "SST4_avg2" target "wpac850_avg2"]
                    edge[source "SST4_avg2" target "cpac850_avg2"]
                    edge[source "SST4_avg2" target "epac850_avg2"]
                    edge[source "SST4_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "NATL_avg2" target "SATL_avg2"]
                    edge[source "NATL_avg2" target "TROP_avg2"]
                    edge[source "NATL_avg2" target "SOI_avg2"]
                    edge[source "NATL_avg2" target "ESOI_avg2"]
                    edge[source "NATL_avg2" target "cpolr_avg2"]
                    edge[source "NATL_avg2" target "wpac850_avg2"]
                    edge[source "NATL_avg2" target "cpac850_avg2"]
                    edge[source "NATL_avg2" target "epac850_avg2"]
                    edge[source "NATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SATL_avg2" target "TROP_avg2"]
                    edge[source "SATL_avg2" target "SOI_avg2"]
                    edge[source "SATL_avg2" target "ESOI_avg2"]
                    edge[source "SATL_avg2" target "cpolr_avg2"]
                    edge[source "SATL_avg2" target "wpac850_avg2"]
                    edge[source "SATL_avg2" target "cpac850_avg2"]
                    edge[source "SATL_avg2" target "epac850_avg2"]
                    edge[source "SATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "TROP_avg2" target "SOI_avg2"]
                    edge[source "TROP_avg2" target "ESOI_avg2"]
                    edge[source "TROP_avg2" target "cpolr_avg2"]
                    edge[source "TROP_avg2" target "wpac850_avg2"]
                    edge[source "TROP_avg2" target "cpac850_avg2"]
                    edge[source "TROP_avg2" target "epac850_avg2"]
                    edge[source "TROP_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SOI_avg2" target "ESOI_avg2"]
                    edge[source "SOI_avg2" target "cpolr_avg2"]
                    edge[source "SOI_avg2" target "wpac850_avg2"]
                    edge[source "SOI_avg2" target "cpac850_avg2"]
                    edge[source "SOI_avg2" target "epac850_avg2"]
                    edge[source "SOI_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "ESOI_avg2" target "cpolr_avg2"]
                    edge[source "ESOI_avg2" target "wpac850_avg2"]
                    edge[source "ESOI_avg2" target "cpac850_avg2"]
                    edge[source "ESOI_avg2" target "epac850_avg2"]
                    edge[source "ESOI_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpolr_avg2" target "wpac850_avg2"]
                    edge[source "cpolr_avg2" target "cpac850_avg2"]
                    edge[source "cpolr_avg2" target "epac850_avg2"]
                    edge[source "cpolr_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "wpac850_avg2" target "cpac850_avg2"]
                    edge[source "wpac850_avg2" target "epac850_avg2"]
                    edge[source "wpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpac850_avg2" target "epac850_avg2"]
                    edge[source "cpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "epac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST3_avg2" target "SST12_avg2"]
                    edge[source "SST3_avg2" target "excess"]
                    edge[source "SST34_avg2" target "SST12_avg2"]
                    edge[source "SST34_avg2" target "excess"]
                    edge[source "SST4_avg2" target "SST12_avg2"]
                    edge[source "SST4_avg2" target "excess"]
                    edge[source "NATL_avg2" target "SST12_avg2"]
                    edge[source "NATL_avg2" target "excess"]
                    edge[source "SATL_avg2" target "SST12_avg2"]
                    edge[source "SATL_avg2" target "excess"]
                    edge[source "TROP_avg2" target "SST12_avg2"]
                    edge[source "TROP_avg2" target "excess"]
                    edge[source "SOI_avg2" target "SST12_avg2"]
                    edge[source "SOI_avg2" target "excess"]
                    edge[source "ESOI_avg2" target "SST12_avg2"]
                    edge[source "ESOI_avg2" target "excess"]
                    edge[source "cpolr_avg2" target "SST12_avg2"]
                    edge[source "cpolr_avg2" target "excess"]
                    edge[source "wpac850_avg2" target "SST12_avg2"]
                    edge[source "wpac850_avg2" target "excess"]
                    edge[source "cpac850_avg2" target "SST12_avg2"]
                    edge[source "cpac850_avg2" target "excess"]
                    edge[source "epac850_avg2" target "SST12_avg2"]
                    edge[source "epac850_avg2" target "excess"]
                    edge[source "qbo_u30_avg2" target "SST12_avg2"]
                    edge[source "qbo_u30_avg2" target "excess"]
                    
                    edge[source "SST3_avg2" target "t2m_avg2"]
                    edge[source "SST3_avg2" target "tp_avg2"]
                    
                    edge[source "SST34_avg2" target "t2m_avg2"]
                    edge[source "SST34_avg2" target "tp_avg2"]
                    
                    edge[source "SST4_avg2" target "t2m_avg2"]
                    edge[source "SST4_avg2" target "tp_avg2"]
                    
                    edge[source "NATL_avg2" target "t2m_avg2"]
                    edge[source "NATL_avg2" target "tp_avg2"]
                    
                    edge[source "SATL_avg2" target "t2m_avg2"]
                    edge[source "SATL_avg2" target "tp_avg2"]
                    
                    edge[source "TROP_avg2" target "t2m_avg2"]
                    edge[source "TROP_avg2" target "tp_avg2"]
                    
                    edge[source "SOI_avg2" target "t2m_avg2"]
                    edge[source "SOI_avg2" target "tp_avg2"]


                    edge[source "ESOI_avg2" target "t2m_avg2"]
                    edge[source "ESOI_avg2" target "tp_avg2"]
                    

                    edge[source "cpolr_avg2" target "t2m_avg2"]
                    edge[source "cpolr_avg2" target "tp_avg2"]
                    
                    edge[source "wpac850_avg2" target "t2m_avg2"]
                    edge[source "wpac850_avg2" target "tp_avg2"]
                    
                    edge[source "cpac850_avg2" target "t2m_avg2"]
                    edge[source "cpac850_avg2" target "tp_avg2"]
                    
                    edge[source "epac850_avg2" target "t2m_avg2"]
                    edge[source "epac850_avg2" target "tp_avg2"]
                    
                    edge[source "qbo_u30_avg2" target "t2m_avg2"]
                    edge[source "qbo_u30_avg2" target "tp_avg2"]
                    
                    edge[source "SST12_avg2" target "t2m_avg2"]
                    edge[source "SST12_avg2" target "tp_avg2"]
                    
                    edge[source "t2m_avg2" target "excess"]
                    edge[source "tp_avg2" target "excess"]           
                    
                    edge[source "t2m_avg2" target "MPI"]
                    edge[source "tp_avg2" target "MPI"]
                    
                    edge[source "t2m_avg2" target "pop_density"]
                    edge[source "tp_avg2" target "pop_density"]                   
                  
                    
                    edge[source "MPI" target "excess"]
                    
                    edge[source "MPI" target "pop_density"]

                    edge[source "pop_density" target "excess"]
                    
                    edge[source "DANE_normalized" target "excess"]
                    edge[source "DANE_year_normalized" target "excess"]
                    edge[source "DANE_period_normalized" target "excess"]
                    
                    edge[source "SST12_avg2" target "excess"]
                    
                    edge[source "altitude" target "t2m_avg2"]
                    
                    
                    node[id "consensus" label "consensus"]
                    edge[source "consensus" target "t2m_avg2"]
                    edge[source "consensus" target "tp_avg2"]
                    edge[source "consensus" target "excess"]
                    edge[source "SOI_avg2" target "consensus"]
                    edge[source "SST4_avg2" target "consensus"]
                    edge[source "SST12_avg2" target "consensus"]
                    edge[source "SST34_avg2" target "consensus"]
                    edge[source "SST3_avg2" target "consensus"]
            
            
                    ]"""
                    )

#%% 

# Identifying effects
identified_estimand_SST12_avg2 = model_SST12.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_SST12_avg2)

#%%

reg1 = lambda: XGBRegressor(n_estimators=5000, random_state=123, eta=0.0001, max_depth=10, reg_lambda=1.5, alpha=0.001)

# Model with DoWhy
estimate_SST12 = model_SST12.estimate_effect(
    identified_estimand_SST12_avg2,
    effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST12_t1'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_outcome":True,
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)


#%%

# ATE
econml_estimator_SST12 = estimate_SST12.estimator.estimator
effect_modifier_cols = ['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST12_t1']
X_data = data[effect_modifier_cols]  
X_data = X_data.dropna()

ate_SST12 = econml_estimator_SST12.effect(
    X=X_data  
).mean()  

print(ate_SST12)

# CI
ate_ci_SST12 = econml_estimator_SST12.effect_interval(
    X=X_data,
    alpha=0.05
)

ci_lower_SST12 = ate_ci_SST12[0].mean()
ci_upper_SST12 = ate_ci_SST12[1].mean()

print(ci_lower_SST12)
print(ci_upper_SST12)

data_ATE.at[0, 'ATE'] = ate_SST12
data_ATE.at[0, 'ci_lower'] = ci_lower_SST12
data_ATE.at[0, 'ci_upper'] = ci_upper_SST12
print(data_ATE)

#%%

effect_SST12 = econml_estimator_SST12.effect(
    X=X_data  
)  

effect_ci_SST12 = econml_estimator_SST12.effect_interval(
    X=X_data,
    alpha=0.05
)

# CATE NATL_avg2
alt = data_SST12_avg2['altitude']  

# Grid for alt
min_alt = alt.min()
max_alt = alt.max()
delta = (max_alt - min_alt) / 100
alt_grid = np.arange(min_alt, max_alt + delta - 0.001, delta)

# Means
DANE_encoded_mean = data_SST12_avg2['DANE_normalized'].mean()
DANE_year_encoded_mean = data_SST12_avg2['DANE_year_normalized'].mean()
DANE_period_encoded_mean = data_SST12_avg2['DANE_period_normalized'].mean()
SST12_t1_mean = data_SST12_avg2['SST12_t1'].mean()

# Matrix for prdiction
X_test_grid = np.column_stack([
    alt_grid,
    np.full_like(alt_grid, DANE_encoded_mean),
    np.full_like(alt_grid, DANE_year_encoded_mean),
    np.full_like(alt_grid, DANE_period_encoded_mean),
    np.full_like(alt_grid, SST12_t1_mean),
])

# Predicction effect
treatment_effect = econml_estimator_SST12.effect(X_test_grid)

hte_lower2_cons, hte_upper2_cons = econml_estimator_SST12.effect_interval(X_test_grid, alpha=0.05)
    
plot_data = pd.DataFrame({
    'alt': alt_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

cate_plot = (
    ggplot(plot_data)
    + aes(x='alt', y='treatment_effect')
    + geom_line(color='black', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Altitude (m)', y='Effect of SST12 on excess dengue cases', title='a')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(
        plot_title=element_text(hjust=0.5, size=12),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10)
    )
)

print(cate_plot)    

#%%

# whit common cause
random_SST12 = model_SST12.refute_estimate(identified_estimand_SST12_avg2, estimate_SST12,
                                         method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_SST12 )

#with subset
subset_SST12  = model_SST12.refute_estimate(identified_estimand_SST12_avg2, estimate_SST12,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_SST12) 
      
#with placebo 
placebo_SST12  = model_SST12.refute_estimate(identified_estimand_SST12_avg2, estimate_SST12,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_SST12)


#%%

data_SST3_avg2 = data[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST12_avg2', 'SST3_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST3_t1']]

scaler = StandardScaler()
data_SST3_avg2['SST3_avg2'] = scaler.fit_transform(data_SST3_avg2[['SST3_avg2']])
data_SST3_avg2['SST3_t1'] = scaler.fit_transform(data_SST3_avg2[['SST3_t1']])
data_SST3_avg2['NATL_avg2'] = scaler.fit_transform(data_SST3_avg2[['NATL_avg2']])

# Convert columns to binary
columns_convert = ['SST12_avg2', 'SST34_avg2', 'SST4_avg2', 'SATL_avg2', 'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'TROP_avg2', 'NATL_avg2']
for col in columns_convert:
    median = data[col].median()
    data[col] = (data[col] > median).astype(int)
    

#%%

# SST3_avg2

data_SST3_avg2 = data_SST3_avg2[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST3_avg2', 'SST12_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST3_t1']]

data_SST3_avg2 = data_SST3_avg2.dropna()

#%%

#Causal mechanism
model_SST3_avg2 = CausalModel(
        data = data_SST3_avg2,
        treatment=['SST3_avg2'],
        outcome=['excess'],
        effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST3_t1'],
        common_causes=['SST12_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                       'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2',
                       'Year', 'sin_month', 'cos_month'], 
        graph= """graph[directed 1 
                    node[id "SST3_avg2" label "SST3_avg2"]
                    node[id "excess" label "excess"]
                    node[id "SST12_avg2" label "SST12_avg2"]
                    node[id "SST34_avg2" label "SST34_avg2"]
                    node[id "SST4_avg2" label "SST4_avg2"]
                    node[id "NATL_avg2" label "NATL_avg2"]
                    node[id "SATL_avg2" label "SATL_avg2"]
                    node[id "TROP_avg2" label "TROP_avg2"]
                    node[id "SOI_avg2" label "SOI_avg2"]
                    node[id "ESOI_avg2" label "ESOI_avg2"]
                    node[id "cpolr_avg2" label "cpolr_avg2"]
                    node[id "wpac850_avg2" label "wpac850_avg2"]
                    node[id "cpac850_avg2" label "cpac850_avg2"]
                    node[id "epac850_avg2" label "epac850_avg2"]
                    node[id "qbo_u30_avg2" label "qbo_u30_avg2"]
                    node[id "t2m_avg2" label "t2m_avg2"]
                    node[id "tp_avg2" label "tp_avg2"]
                    node[id "MPI" label "MPI"]
                    node[id "pop_density" label "pop_density"]
                    node[id "Year" label "Year"]
                    node[id "sin_month" label "sin_month"]
                    node[id "cos_month" label "cos_month"]
                    node[id "DANE_normalized" label "DANE_normalized"]
                    node[id "DANE_year_normalized" label "DANE_year_normalized"]
                    node[id "DANE_period_normalized" label "DANE_period_normalized"]
                    node[id "altitude" label "altitude"]

                    
                    edge[source "Year" target "SST12_avg2"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3_avg2"]
                    edge[source "Year" target "SST34_avg2"]
                    edge[source "Year" target "SST4_avg2"]
                    edge[source "Year" target "NATL_avg2"]
                    edge[source "Year" target "SATL_avg2"]
                    edge[source "Year" target "TROP_avg2"]
                    edge[source "Year" target "SOI_avg2"]
                    edge[source "Year" target "ESOI_avg2"]
                    edge[source "Year" target "cpolr_avg2"]
                    edge[source "Year" target "wpac850_avg2"]
                    edge[source "Year" target "cpac850_avg2"]
                    edge[source "Year" target "epac850_avg2"]
                    edge[source "Year" target "qbo_u30_avg2"]
                    edge[source "Year" target "t2m_avg2"]
                    edge[source "Year" target "tp_avg2"]
                    
        		    edge[source "sin_month" target "SST3_avg2"]
                    edge[source "sin_month" target "excess"]
                    edge[source "sin_month" target "SST12_avg2"]
                    edge[source "sin_month" target "SST34_avg2"]
                    edge[source "sin_month" target "SST4_avg2"]
                    edge[source "sin_month" target "NATL_avg2"]
                    edge[source "sin_month" target "SATL_avg2"]
                    edge[source "sin_month" target "TROP_avg2"]
                    edge[source "sin_month" target "SOI_avg2"]
                    edge[source "sin_month" target "ESOI_avg2"]
                    edge[source "sin_month" target "cpolr_avg2"]
                    edge[source "sin_month" target "wpac850_avg2"]
                    edge[source "sin_month" target "cpac850_avg2"]
                    edge[source "sin_month" target "epac850_avg2"]
                    edge[source "sin_month" target "qbo_u30_avg2"]
                    edge[source "sin_month" target "t2m_avg2"]
                    edge[source "sin_month" target "tp_avg2"]

                    edge[source "cos_month" target "SST12_avg2"]
                    edge[source "cos_month" target "excess"]
                    edge[source "cos_month" target "SST3_avg2"]
                    edge[source "cos_month" target "SST34_avg2"]
                    edge[source "cos_month" target "SST4_avg2"]
                    edge[source "cos_month" target "NATL_avg2"]
                    edge[source "cos_month" target "SATL_avg2"]
                    edge[source "cos_month" target "TROP_avg2"]
                    edge[source "cos_month" target "SOI_avg2"]
                    edge[source "cos_month" target "ESOI_avg2"]
                    edge[source "cos_month" target "cpolr_avg2"]
                    edge[source "cos_month" target "wpac850_avg2"]
                    edge[source "cos_month" target "cpac850_avg2"]
                    edge[source "cos_month" target "epac850_avg2"]
                    edge[source "cos_month" target "qbo_u30_avg2"]
                    edge[source "cos_month" target "t2m_avg2"]
                    edge[source "cos_month" target "tp_avg2"]
                    
                           
                    edge[source "SST12_avg2" target "NATL_avg2"]
                    edge[source "SST12_avg2" target "SATL_avg2"]
                    edge[source "SST12_avg2" target "TROP_avg2"]
                    edge[source "SST12_avg2" target "SOI_avg2"]
                    edge[source "SST12_avg2" target "ESOI_avg2"]
                    edge[source "SST12_avg2" target "cpolr_avg2"]
                    edge[source "SST12_avg2" target "wpac850_avg2"]
                    edge[source "SST12_avg2" target "cpac850_avg2"]
                    edge[source "SST12_avg2" target "epac850_avg2"]
                    edge[source "SST12_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST34_avg2" target "SST4_avg2"]
                    edge[source "SST34_avg2" target "NATL_avg2"]
                    edge[source "SST34_avg2" target "SATL_avg2"]
                    edge[source "SST34_avg2" target "TROP_avg2"]
                    edge[source "SST34_avg2" target "SOI_avg2"]
                    edge[source "SST34_avg2" target "ESOI_avg2"]
                    edge[source "SST34_avg2" target "cpolr_avg2"]
                    edge[source "SST34_avg2" target "wpac850_avg2"]
                    edge[source "SST34_avg2" target "cpac850_avg2"]
                    edge[source "SST34_avg2" target "epac850_avg2"]
                    edge[source "SST34_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST4_avg2" target "NATL_avg2"]
                    edge[source "SST4_avg2" target "SATL_avg2"]
                    edge[source "SST4_avg2" target "TROP_avg2"]
                    edge[source "SST4_avg2" target "SOI_avg2"]
                    edge[source "SST4_avg2" target "ESOI_avg2"]
                    edge[source "SST4_avg2" target "cpolr_avg2"]
                    edge[source "SST4_avg2" target "wpac850_avg2"]
                    edge[source "SST4_avg2" target "cpac850_avg2"]
                    edge[source "SST4_avg2" target "epac850_avg2"]
                    edge[source "SST4_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "NATL_avg2" target "SATL_avg2"]
                    edge[source "NATL_avg2" target "TROP_avg2"]
                    edge[source "NATL_avg2" target "SOI_avg2"]
                    edge[source "NATL_avg2" target "ESOI_avg2"]
                    edge[source "NATL_avg2" target "cpolr_avg2"]
                    edge[source "NATL_avg2" target "wpac850_avg2"]
                    edge[source "NATL_avg2" target "cpac850_avg2"]
                    edge[source "NATL_avg2" target "epac850_avg2"]
                    edge[source "NATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SATL_avg2" target "TROP_avg2"]
                    edge[source "SATL_avg2" target "SOI_avg2"]
                    edge[source "SATL_avg2" target "ESOI_avg2"]
                    edge[source "SATL_avg2" target "cpolr_avg2"]
                    edge[source "SATL_avg2" target "wpac850_avg2"]
                    edge[source "SATL_avg2" target "cpac850_avg2"]
                    edge[source "SATL_avg2" target "epac850_avg2"]
                    edge[source "SATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "TROP_avg2" target "SOI_avg2"]
                    edge[source "TROP_avg2" target "ESOI_avg2"]
                    edge[source "TROP_avg2" target "cpolr_avg2"]
                    edge[source "TROP_avg2" target "wpac850_avg2"]
                    edge[source "TROP_avg2" target "cpac850_avg2"]
                    edge[source "TROP_avg2" target "epac850_avg2"]
                    edge[source "TROP_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SOI_avg2" target "ESOI_avg2"]
                    edge[source "SOI_avg2" target "cpolr_avg2"]
                    edge[source "SOI_avg2" target "wpac850_avg2"]
                    edge[source "SOI_avg2" target "cpac850_avg2"]
                    edge[source "SOI_avg2" target "epac850_avg2"]
                    edge[source "SOI_avg2" target "qbo_u30_avg2"]


                    edge[source "ESOI_avg2" target "cpolr_avg2"]
                    edge[source "ESOI_avg2" target "wpac850_avg2"]
                    edge[source "ESOI_avg2" target "cpac850_avg2"]
                    edge[source "ESOI_avg2" target "epac850_avg2"]
                    edge[source "ESOI_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpolr_avg2" target "wpac850_avg2"]
                    edge[source "cpolr_avg2" target "cpac850_avg2"]
                    edge[source "cpolr_avg2" target "epac850_avg2"]
                    edge[source "cpolr_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "wpac850_avg2" target "cpac850_avg2"]
                    edge[source "wpac850_avg2" target "epac850_avg2"]
                    edge[source "wpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpac850_avg2" target "epac850_avg2"]
                    edge[source "cpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "epac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST12_avg2" target "SST3_avg2"]
                    edge[source "SST12_avg2" target "excess"]
                    edge[source "SST34_avg2" target "SST3_avg2"]
                    edge[source "SST34_avg2" target "excess"]
                    edge[source "SST4_avg2" target "SST3_avg2"]
                    edge[source "SST4_avg2" target "excess"]
                    edge[source "NATL_avg2" target "SST3_avg2"]
                    edge[source "NATL_avg2" target "excess"]
                    edge[source "SATL_avg2" target "SST3_avg2"]
                    edge[source "SATL_avg2" target "excess"]
                    edge[source "TROP_avg2" target "SST3_avg2"]
                    edge[source "TROP_avg2" target "excess"]
                    edge[source "SOI_avg2" target "SST3_avg2"]
                    edge[source "SOI_avg2" target "excess"]
                    edge[source "ESOI_avg2" target "SST3_avg2"]
                    edge[source "ESOI_avg2" target "excess"]
                    edge[source "cpolr_avg2" target "SST3_avg2"]
                    edge[source "cpolr_avg2" target "excess"]
                    edge[source "wpac850_avg2" target "SST3_avg2"]
                    edge[source "wpac850_avg2" target "excess"]
                    edge[source "cpac850_avg2" target "SST3_avg2"]
                    edge[source "cpac850_avg2" target "excess"]
                    edge[source "epac850_avg2" target "SST3_avg2"]
                    edge[source "epac850_avg2" target "excess"]
                    edge[source "qbo_u30_avg2" target "SST3_avg2"]
                    edge[source "qbo_u30_avg2" target "excess"]
                    
                    edge[source "SST12_avg2" target "t2m_avg2"]
                    edge[source "SST12_avg2" target "tp_avg2"]
                    
                    edge[source "SST34_avg2" target "t2m_avg2"]
                    edge[source "SST34_avg2" target "tp_avg2"]
                    
                    edge[source "SST4_avg2" target "t2m_avg2"]
                    edge[source "SST4_avg2" target "tp_avg2"]
                    
                    edge[source "NATL_avg2" target "t2m_avg2"]
                    edge[source "NATL_avg2" target "tp_avg2"]
                    
                    edge[source "SATL_avg2" target "t2m_avg2"]
                    edge[source "SATL_avg2" target "tp_avg2"]
                    
                    edge[source "TROP_avg2" target "t2m_avg2"]
                    edge[source "TROP_avg2" target "tp_avg2"]
                    
                    edge[source "SOI_avg2" target "t2m_avg2"]
                    edge[source "SOI_avg2" target "tp_avg2"]


                    edge[source "ESOI_avg2" target "t2m_avg2"]
                    edge[source "ESOI_avg2" target "tp_avg2"]
                    

                    edge[source "cpolr_avg2" target "t2m_avg2"]
                    edge[source "cpolr_avg2" target "tp_avg2"]
                    
                    edge[source "wpac850_avg2" target "t2m_avg2"]
                    edge[source "wpac850_avg2" target "tp_avg2"]
                    
                    edge[source "cpac850_avg2" target "t2m_avg2"]
                    edge[source "cpac850_avg2" target "tp_avg2"]
                    
                    edge[source "epac850_avg2" target "t2m_avg2"]
                    edge[source "epac850_avg2" target "tp_avg2"]
                    
                    edge[source "qbo_u30_avg2" target "t2m_avg2"]
                    edge[source "qbo_u30_avg2" target "tp_avg2"]
                    
                    edge[source "SST3_avg2" target "t2m_avg2"]
                    edge[source "SST3_avg2" target "tp_avg2"]
                    
                    
                    edge[source "t2m_avg2" target "excess"]
                    edge[source "tp_avg2" target "excess"]           
                    
                    edge[source "t2m_avg2" target "MPI"]
                    edge[source "tp_avg2" target "MPI"]
                    
                    edge[source "t2m_avg2" target "pop_density"]
                    edge[source "tp_avg2" target "pop_density"]                   
                  
                    
                    edge[source "MPI" target "excess"]
                    
                    edge[source "MPI" target "pop_density"]

                    edge[source "pop_density" target "excess"]
                    
                    edge[source "DANE_normalized" target "excess"]
                    edge[source "DANE_year_normalized" target "excess"]
                    edge[source "DANE_period_normalized" target "excess"]
                    
                    edge[source "SST3_avg2" target "excess"]
                    
                    edge[source "altitude" target "t2m_avg2"]
                    
                    
                    node[id "consensus" label "consensus"]
                    edge[source "consensus" target "t2m_avg2"]
                    edge[source "consensus" target "tp_avg2"]
                    edge[source "consensus" target "excess"]
                    edge[source "SOI_avg2" target "consensus"]
                    edge[source "SST4_avg2" target "consensus"]
                    edge[source "SST12_avg2" target "consensus"]
                    edge[source "SST34_avg2" target "consensus"]
                    edge[source "SST3_avg2" target "consensus"]
            
            
                    ]"""
                    )

#%% 

# Identifying effects
identified_estimand_SST3_avg2 = model_SST3_avg2.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_SST3_avg2)


#%%


# Model with DoWhy
estimate_SST3_avg2 = model_SST3_avg2.estimate_effect(
    identified_estimand_SST3_avg2,
    effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST3_t1'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_outcome":True,
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)


#%%

# ATE

econml_estimator_SST3_avg2 = estimate_SST3_avg2.estimator.estimator
effect_modifier_cols = ['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST3_t1']
X_data = data[effect_modifier_cols]  
X_data = X_data.dropna()



ate_SST3_avg2 = econml_estimator_SST3_avg2.effect(
    X=X_data 
).mean()  

print(ate_SST3_avg2)

# CI
ate_ci_SST3_avg2 = econml_estimator_SST3_avg2.effect_interval(
    #T=1,
    X=X_data,
    alpha=0.05
)

ci_lower_SST3_avg2 = ate_ci_SST3_avg2[0].mean()
ci_upper_SST3_avg2 = ate_ci_SST3_avg2[1].mean()

print(ci_lower_SST3_avg2)
print(ci_upper_SST3_avg2)

data_ATE.at[1, 'ATE'] = ate_SST3_avg2
data_ATE.at[1, 'ci_lower'] = ci_lower_SST3_avg2
data_ATE.at[1, 'ci_upper'] = ci_upper_SST3_avg2
print(data_ATE)

#%%

effect_SST3_avg2 = econml_estimator_SST3_avg2.effect(
    X=X_data  
)  

effect_ci_SST3_avg2 = econml_estimator_SST3_avg2.effect_interval(
    #T=1,
    X=X_data,
    alpha=0.05
)

# CATE alt
alt = data_SST3_avg2['altitude']  

# Grid for alt
min_alt = alt.min()
max_alt = alt.max()
delta = (max_alt - min_alt) / 100
alt_grid = np.arange(min_alt, max_alt + delta - 0.001, delta)

# Means
DANE_encoded_mean = data_SST3_avg2['DANE_normalized'].mean()
DANE_year_encoded_mean = data_SST3_avg2['DANE_year_normalized'].mean()
DANE_period_encoded_mean = data_SST3_avg2['DANE_period_normalized'].mean()
SST3_t1_mean = data_SST3_avg2['SST3_t1'].mean()

# Matrix for prdiction
X_test_grid = np.column_stack([
    alt_grid,
    np.full_like(alt_grid, DANE_encoded_mean),
    np.full_like(alt_grid, DANE_year_encoded_mean),
    np.full_like(alt_grid, DANE_period_encoded_mean),
    np.full_like(alt_grid, SST3_t1_mean),
])

# Predicction effect
treatment_effect = econml_estimator_SST3_avg2.effect(X_test_grid)

#ci
hte_lower2_cons, hte_upper2_cons = econml_estimator_SST3_avg2.effect_interval(X_test_grid, alpha=0.05)
    
plot_data = pd.DataFrame({
    'alt': alt_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

cate_plot = (
    ggplot(plot_data)
    + aes(x='alt', y='treatment_effect')
    + geom_line(color='black', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Altitude (m)', y='Effect of SST3 on excess dengue cases', title='b')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(
        plot_title=element_text(hjust=0.5, size=12),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10)
    )
)

print(cate_plot)    

#%%

#with common cause
random_SST3_avg2 = model_SST3_avg2.refute_estimate(identified_estimand_SST3_avg2, estimate_SST3_avg2,
                                         method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_SST3_avg2 )

#with subset
subset_SST3_avg2  = model_SST3_avg2.refute_estimate(identified_estimand_SST3_avg2, estimate_SST3_avg2,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_SST3_avg2) 
      
#with placebo 
placebo_SST3_avg2  = model_SST3_avg2.refute_estimate(identified_estimand_SST3_avg2, estimate_SST3_avg2,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_SST3_avg2)
  

#%%

data_SST34_avg2 = data[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST12_avg2', 'SST3_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST34_t1']]

scaler = StandardScaler()
data_SST34_avg2['SST34_avg2'] = scaler.fit_transform(data_SST34_avg2[['SST34_avg2']])
data_SST34_avg2['SST34_t1'] = scaler.fit_transform(data_SST34_avg2[['SST34_t1']])
data_SST34_avg2['NATL_avg2'] = scaler.fit_transform(data_SST34_avg2[['NATL_avg2']])

# Convert columns to binary
columns_convert = ['SST4_avg2', 'SST3_avg2', 'SST12_avg2', 'SATL_avg2', 'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'TROP_avg2', 'NATL_avg2']
for col in columns_convert:
    median = data[col].median()
    data[col] = (data[col] > median).astype(int)


# SST34_avg2
data_SST34_avg2 = data_SST34_avg2[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST34_avg2', 'SST12_avg2', 'SST3_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST34_t1']]

data_SST34_avg2 = data_SST34_avg2.dropna()

#%%

#Causal mechanism
model_SST34_avg2 = CausalModel(
        data = data_SST34_avg2,
        treatment=['SST34_avg2'],
        outcome=['excess'],
        effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST34_t1'],
        common_causes=['SST12_avg2', 'SST3_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                       'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2',
                       'Year', 'sin_month', 'cos_month'], 
        graph= """graph[directed 1 
                    node[id "SST34_avg2" label "SST34_avg2"]
                    node[id "excess" label "excess"]
                    node[id "SST12_avg2" label "SST12_avg2"]
                    node[id "SST3_avg2" label "SST3_avg2"]
                    node[id "SST4_avg2" label "SST4_avg2"]
                    node[id "NATL_avg2" label "NATL_avg2"]
                    node[id "SATL_avg2" label "SATL_avg2"]
                    node[id "TROP_avg2" label "TROP_avg2"]
                    node[id "SOI_avg2" label "SOI_avg2"]
                    node[id "ESOI_avg2" label "ESOI_avg2"]
                    node[id "cpolr_avg2" label "cpolr_avg2"]
                    node[id "wpac850_avg2" label "wpac850_avg2"]
                    node[id "cpac850_avg2" label "cpac850_avg2"]
                    node[id "epac850_avg2" label "epac850_avg2"]
                    node[id "qbo_u30_avg2" label "qbo_u30_avg2"]
                    node[id "t2m_avg2" label "t2m_avg2"]
                    node[id "tp_avg2" label "tp_avg2"]
                    node[id "MPI" label "MPI"]
                    node[id "pop_density" label "pop_density"]
                    node[id "Year" label "Year"]
                    node[id "sin_month" label "sin_month"]
                    node[id "cos_month" label "cos_month"]
                    node[id "DANE_normalized" label "DANE_normalized"]
                    node[id "DANE_year_normalized" label "DANE_year_normalized"]
                    node[id "DANE_period_normalized" label "DANE_period_normalized"]
                    node[id "altitude" label "altitude"]

                    
                    edge[source "Year" target "SST12_avg2"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST34_avg2"]
                    edge[source "Year" target "SST3_avg2"]
                    edge[source "Year" target "SST4_avg2"]
                    edge[source "Year" target "NATL_avg2"]
                    edge[source "Year" target "SATL_avg2"]
                    edge[source "Year" target "TROP_avg2"]
                    edge[source "Year" target "SOI_avg2"]
                    edge[source "Year" target "ESOI_avg2"]
                    edge[source "Year" target "cpolr_avg2"]
                    edge[source "Year" target "wpac850_avg2"]
                    edge[source "Year" target "cpac850_avg2"]
                    edge[source "Year" target "epac850_avg2"]
                    edge[source "Year" target "qbo_u30_avg2"]
                    edge[source "Year" target "t2m_avg2"]
                    edge[source "Year" target "tp_avg2"]
                    
        		    edge[source "sin_month" target "SST12_avg2"]
                    edge[source "sin_month" target "excess"]
                    edge[source "sin_month" target "SST3_avg2"]
                    edge[source "sin_month" target "SST34_avg2"]
                    edge[source "sin_month" target "SST4_avg2"]
                    edge[source "sin_month" target "NATL_avg2"]
                    edge[source "sin_month" target "SATL_avg2"]
                    edge[source "sin_month" target "TROP_avg2"]
                    edge[source "sin_month" target "SOI_avg2"]
                    edge[source "sin_month" target "ESOI_avg2"]
                    edge[source "sin_month" target "cpolr_avg2"]
                    edge[source "sin_month" target "wpac850_avg2"]
                    edge[source "sin_month" target "cpac850_avg2"]
                    edge[source "sin_month" target "epac850_avg2"]
                    edge[source "sin_month" target "qbo_u30_avg2"]
                    edge[source "sin_month" target "t2m_avg2"]
                    edge[source "sin_month" target "tp_avg2"]

                    edge[source "cos_month" target "SST12_avg2"]
                    edge[source "cos_month" target "excess"]
                    edge[source "cos_month" target "SST3_avg2"]
                    edge[source "cos_month" target "SST34_avg2"]
                    edge[source "cos_month" target "SST4_avg2"]
                    edge[source "cos_month" target "NATL_avg2"]
                    edge[source "cos_month" target "SATL_avg2"]
                    edge[source "cos_month" target "TROP_avg2"]
                    edge[source "cos_month" target "SOI_avg2"]
                    edge[source "cos_month" target "ESOI_avg2"]
                    edge[source "cos_month" target "cpolr_avg2"]
                    edge[source "cos_month" target "wpac850_avg2"]
                    edge[source "cos_month" target "cpac850_avg2"]
                    edge[source "cos_month" target "epac850_avg2"]
                    edge[source "cos_month" target "qbo_u30_avg2"]
                    edge[source "cos_month" target "t2m_avg2"]
                    edge[source "cos_month" target "tp_avg2"]
                    
                           
                    edge[source "SST12_avg2" target "SST3_avg2"]
                    edge[source "SST12_avg2" target "SST4_avg2"]
                    edge[source "SST12_avg2" target "NATL_avg2"]
                    edge[source "SST12_avg2" target "SATL_avg2"]
                    edge[source "SST12_avg2" target "TROP_avg2"]
                    edge[source "SST12_avg2" target "SOI_avg2"]
                    edge[source "SST12_avg2" target "ESOI_avg2"]
                    edge[source "SST12_avg2" target "cpolr_avg2"]
                    edge[source "SST12_avg2" target "wpac850_avg2"]
                    edge[source "SST12_avg2" target "cpac850_avg2"]
                    edge[source "SST12_avg2" target "epac850_avg2"]
                    edge[source "SST12_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST3_avg2" target "SST4_avg2"]
                    edge[source "SST3_avg2" target "NATL_avg2"]
                    edge[source "SST3_avg2" target "SATL_avg2"]
                    edge[source "SST3_avg2" target "TROP_avg2"]
                    edge[source "SST3_avg2" target "SOI_avg2"]
                    edge[source "SST3_avg2" target "ESOI_avg2"]
                    edge[source "SST3_avg2" target "cpolr_avg2"]
                    edge[source "SST3_avg2" target "wpac850_avg2"]
                    edge[source "SST3_avg2" target "cpac850_avg2"]
                    edge[source "SST3_avg2" target "epac850_avg2"]
                    edge[source "SST3_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST4_avg2" target "NATL_avg2"]
                    edge[source "SST4_avg2" target "SATL_avg2"]
                    edge[source "SST4_avg2" target "TROP_avg2"]
                    edge[source "SST4_avg2" target "SOI_avg2"]
                    edge[source "SST4_avg2" target "ESOI_avg2"]
                    edge[source "SST4_avg2" target "cpolr_avg2"]
                    edge[source "SST4_avg2" target "wpac850_avg2"]
                    edge[source "SST4_avg2" target "cpac850_avg2"]
                    edge[source "SST4_avg2" target "epac850_avg2"]
                    edge[source "SST4_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "NATL_avg2" target "SATL_avg2"]
                    edge[source "NATL_avg2" target "TROP_avg2"]
                    edge[source "NATL_avg2" target "SOI_avg2"]
                    edge[source "NATL_avg2" target "ESOI_avg2"]
                    edge[source "NATL_avg2" target "cpolr_avg2"]
                    edge[source "NATL_avg2" target "wpac850_avg2"]
                    edge[source "NATL_avg2" target "cpac850_avg2"]
                    edge[source "NATL_avg2" target "epac850_avg2"]
                    edge[source "NATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SATL_avg2" target "TROP_avg2"]
                    edge[source "SATL_avg2" target "SOI_avg2"]
                    edge[source "SATL_avg2" target "ESOI_avg2"]
                    edge[source "SATL_avg2" target "cpolr_avg2"]
                    edge[source "SATL_avg2" target "wpac850_avg2"]
                    edge[source "SATL_avg2" target "cpac850_avg2"]
                    edge[source "SATL_avg2" target "epac850_avg2"]
                    edge[source "SATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "TROP_avg2" target "SOI_avg2"]
                    edge[source "TROP_avg2" target "ESOI_avg2"]
                    edge[source "TROP_avg2" target "cpolr_avg2"]
                    edge[source "TROP_avg2" target "wpac850_avg2"]
                    edge[source "TROP_avg2" target "cpac850_avg2"]
                    edge[source "TROP_avg2" target "epac850_avg2"]
                    edge[source "TROP_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SOI_avg2" target "ESOI_avg2"]
                    edge[source "SOI_avg2" target "cpolr_avg2"]
                    edge[source "SOI_avg2" target "wpac850_avg2"]
                    edge[source "SOI_avg2" target "cpac850_avg2"]
                    edge[source "SOI_avg2" target "epac850_avg2"]
                    edge[source "SOI_avg2" target "qbo_u30_avg2"]


                    edge[source "ESOI_avg2" target "cpolr_avg2"]
                    edge[source "ESOI_avg2" target "wpac850_avg2"]
                    edge[source "ESOI_avg2" target "cpac850_avg2"]
                    edge[source "ESOI_avg2" target "epac850_avg2"]
                    edge[source "ESOI_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpolr_avg2" target "wpac850_avg2"]
                    edge[source "cpolr_avg2" target "cpac850_avg2"]
                    edge[source "cpolr_avg2" target "epac850_avg2"]
                    edge[source "cpolr_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "wpac850_avg2" target "cpac850_avg2"]
                    edge[source "wpac850_avg2" target "epac850_avg2"]
                    edge[source "wpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpac850_avg2" target "epac850_avg2"]
                    edge[source "cpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "epac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST12_avg2" target "SST34_avg2"]
                    edge[source "SST12_avg2" target "excess"]
                    edge[source "SST3_avg2" target "SST34_avg2"]
                    edge[source "SST3_avg2" target "excess"]
                    edge[source "SST4_avg2" target "SST34_avg2"]
                    edge[source "SST4_avg2" target "excess"]
                    edge[source "NATL_avg2" target "SST34_avg2"]
                    edge[source "NATL_avg2" target "excess"]
                    edge[source "SATL_avg2" target "SST34_avg2"]
                    edge[source "SATL_avg2" target "excess"]
                    edge[source "TROP_avg2" target "SST34_avg2"]
                    edge[source "TROP_avg2" target "excess"]
                    edge[source "SOI_avg2" target "SST34_avg2"]
                    edge[source "SOI_avg2" target "excess"]
                    edge[source "ESOI_avg2" target "SST34_avg2"]
                    edge[source "ESOI_avg2" target "excess"]
                    edge[source "cpolr_avg2" target "SST34_avg2"]
                    edge[source "cpolr_avg2" target "excess"]
                    edge[source "wpac850_avg2" target "SST34_avg2"]
                    edge[source "wpac850_avg2" target "excess"]
                    edge[source "cpac850_avg2" target "SST34_avg2"]
                    edge[source "cpac850_avg2" target "excess"]
                    edge[source "epac850_avg2" target "SST34_avg2"]
                    edge[source "epac850_avg2" target "excess"]
                    edge[source "qbo_u30_avg2" target "SST34_avg2"]
                    edge[source "qbo_u30_avg2" target "excess"]
                    
                    edge[source "SST12_avg2" target "t2m_avg2"]
                    edge[source "SST12_avg2" target "tp_avg2"]
                    
                    edge[source "SST3_avg2" target "t2m_avg2"]
                    edge[source "SST3_avg2" target "tp_avg2"]
                    
                    edge[source "SST4_avg2" target "t2m_avg2"]
                    edge[source "SST4_avg2" target "tp_avg2"]
                    
                    edge[source "NATL_avg2" target "t2m_avg2"]
                    edge[source "NATL_avg2" target "tp_avg2"]
                    
                    edge[source "SATL_avg2" target "t2m_avg2"]
                    edge[source "SATL_avg2" target "tp_avg2"]
                    
                    edge[source "TROP_avg2" target "t2m_avg2"]
                    edge[source "TROP_avg2" target "tp_avg2"]
                    
                    edge[source "SOI_avg2" target "t2m_avg2"]
                    edge[source "SOI_avg2" target "tp_avg2"]


                    edge[source "ESOI_avg2" target "t2m_avg2"]
                    edge[source "ESOI_avg2" target "tp_avg2"]
                    

                    edge[source "cpolr_avg2" target "t2m_avg2"]
                    edge[source "cpolr_avg2" target "tp_avg2"]
                    
                    edge[source "wpac850_avg2" target "t2m_avg2"]
                    edge[source "wpac850_avg2" target "tp_avg2"]
                    
                    edge[source "cpac850_avg2" target "t2m_avg2"]
                    edge[source "cpac850_avg2" target "tp_avg2"]
                    
                    edge[source "epac850_avg2" target "t2m_avg2"]
                    edge[source "epac850_avg2" target "tp_avg2"]
                    
                    edge[source "qbo_u30_avg2" target "t2m_avg2"]
                    edge[source "qbo_u30_avg2" target "tp_avg2"]
                    
                    edge[source "SST34_avg2" target "t2m_avg2"]
                    edge[source "SST34_avg2" target "tp_avg2"]
                    
                    
                    edge[source "t2m_avg2" target "excess"]
                    edge[source "tp_avg2" target "excess"]           
                    
                    edge[source "t2m_avg2" target "MPI"]
                    edge[source "tp_avg2" target "MPI"]
                    
                    edge[source "t2m_avg2" target "pop_density"]
                    edge[source "tp_avg2" target "pop_density"]                   
                  
                    
                    edge[source "MPI" target "excess"]
                    
                    edge[source "MPI" target "pop_density"]

                    edge[source "pop_density" target "excess"]
                    
                    edge[source "DANE_normalized" target "excess"]
                    edge[source "DANE_year_normalized" target "excess"]
                    edge[source "DANE_period_normalized" target "excess"]
                    
                    edge[source "SST34_avg2" target "excess"]
                    
                    edge[source "altitude" target "t2m_avg2"]
                    
                    
                    node[id "consensus" label "consensus"]
                    edge[source "consensus" target "t2m_avg2"]
                    edge[source "consensus" target "tp_avg2"]
                    edge[source "consensus" target "excess"]
                    edge[source "SOI_avg2" target "consensus"]
                    edge[source "SST4_avg2" target "consensus"]
                    edge[source "SST12_avg2" target "consensus"]
                    edge[source "SST34_avg2" target "consensus"]
                    edge[source "SST3_avg2" target "consensus"]
            
                    ]"""
                    )

#%% 

# Identifying effects
identified_estimand_SST34_avg2 = model_SST34_avg2.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_SST34_avg2)


#%%


# Model with DoWhy
estimate_SST34_avg2 = model_SST34_avg2.estimate_effect(
    identified_estimand_SST34_avg2,
    effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST34_t1'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_outcome":True,
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)


#%%

# ATE

econml_estimator_SST34_avg2 = estimate_SST34_avg2.estimator.estimator
effect_modifier_cols = ['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST34_t1']
X_data = data[effect_modifier_cols] 
X_data = X_data.dropna()



ate_SST34_avg2 = econml_estimator_SST34_avg2.effect(
    X=X_data 
).mean()  

print(ate_SST34_avg2)

# CI
ate_ci_SST34_avg2 = econml_estimator_SST34_avg2.effect_interval(
    X=X_data,
    alpha=0.05
)

ci_lower_SST34_avg2 = ate_ci_SST34_avg2[0].mean()
ci_upper_SST34_avg2 = ate_ci_SST34_avg2[1].mean()

print(ci_lower_SST34_avg2)
print(ci_upper_SST34_avg2)

data_ATE.at[2, 'ATE'] = ate_SST34_avg2
data_ATE.at[2, 'ci_lower'] = ci_lower_SST34_avg2
data_ATE.at[2, 'ci_upper'] = ci_upper_SST34_avg2
print(data_ATE)

data_ATE.to_csv("D:/data_ATE.csv")

#%%

effect_SST34_avg2 = econml_estimator_SST34_avg2.effect(
    X=X_data  
) 

# CI
effect_ci_SST34_avg2 = econml_estimator_SST34_avg2.effect_interval(
    X=X_data,
    alpha=0.05
)

# CATE alt
alt = data_SST34_avg2['altitude'] 

# Grid for alt
min_alt = alt.min()
max_alt = alt.max()
delta = (max_alt - min_alt) / 100
alt_grid = np.arange(min_alt, max_alt + delta - 0.001, delta)

# Means
DANE_encoded_mean = data_SST34_avg2['DANE_normalized'].mean()
DANE_year_encoded_mean = data_SST34_avg2['DANE_year_normalized'].mean()
DANE_period_encoded_mean = data_SST34_avg2['DANE_period_normalized'].mean()
SST34_t1_mean = data_SST34_avg2['SST34_t1'].mean()

# Matrix for prdiction
X_test_grid = np.column_stack([
    alt_grid,
    np.full_like(alt_grid, DANE_encoded_mean),
    np.full_like(alt_grid, DANE_year_encoded_mean),
    np.full_like(alt_grid, DANE_period_encoded_mean),
    np.full_like(alt_grid, SST34_t1_mean),
])

# Predicction effect
treatment_effect = econml_estimator_SST34_avg2.effect(X_test_grid)

hte_lower2_cons, hte_upper2_cons = econml_estimator_SST34_avg2.effect_interval(X_test_grid, alpha=0.05)
    
plot_data = pd.DataFrame({
    'alt': alt_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

cate_plot = (
    ggplot(plot_data)
    + aes(x='alt', y='treatment_effect')
    + geom_line(color='black', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Altitude (m)', y='Effect of SST34 on excess dengue cases', title='c')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(
        plot_title=element_text(hjust=0.5, size=12),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10)
    )
)

print(cate_plot)    

#%%

#whit common cause
random_SST34_avg2 = model_SST34_avg2.refute_estimate(identified_estimand_SST34_avg2, estimate_SST34_avg2,
                                         method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_SST34_avg2)

#with subset
subset_SST34_avg2  = model_SST34_avg2.refute_estimate(identified_estimand_SST34_avg2, estimate_SST34_avg2,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_SST34_avg2) 
      
#with placebo 
placebo_SST34_avg2  = model_SST34_avg2.refute_estimate(identified_estimand_SST34_avg2, estimate_SST34_avg2,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_SST34_avg2)

#%%

data_SST4_avg2 = data[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST12_avg2', 'SST3_avg2', 'SST34_avg2', 'SST4_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST4_t1']]

scaler = StandardScaler()
data_SST4_avg2['SST4_avg2'] = scaler.fit_transform(data_SST4_avg2[['SST4_avg2']])
data_SST4_avg2['SST4_t1'] = scaler.fit_transform(data_SST4_avg2[['SST4_t1']])
data_SST4_avg2['NATL_avg2'] = scaler.fit_transform(data_SST4_avg2[['NATL_avg2']])

# Convert columns to binary
columns_convert = ['SST3_avg2', 'SST34_avg2', 'SST12_avg2', 'SATL_avg2', 'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'TROP_avg2','NATL_avg2']
for col in columns_convert:
    median = data[col].median()
    data[col] = (data[col] > median).astype(int)


# SST4_avg2
data_SST4_avg2 = data_SST4_avg2[['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month',
                      'SST4_avg2', 'SST3_avg2', 'SST34_avg2', 'SST12_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                      'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2', 'altitude', 'consensus',
                      't2m_avg2', 'tp_avg2', 'MPI', 'pop_density', 'excess', 'SST4_t1']]

data_SST4_avg2 = data_SST4_avg2.dropna()

#%%

#Causal mechanism
model_SST4_avg2 = CausalModel(
        data = data_SST4_avg2,
        treatment=['SST4_avg2'],
        outcome=['excess'],
        effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST4_t1'],
        common_causes=['SST3_avg2', 'SST34_avg2', 'SST12_avg2', 'NATL_avg2', 'SATL_avg2', 'TROP_avg2', 
                       'SOI_avg2', 'ESOI_avg2', 'cpolr_avg2', 'wpac850_avg2', 'cpac850_avg2', 'epac850_avg2', 'qbo_u30_avg2',
                       'Year', 'sin_month', 'cos_month'], 
        graph= """graph[directed 1 
                    node[id "SST4_avg2" label "SST4_avg2"]
                    node[id "excess" label "excess"]
                    node[id "SST3_avg2" label "SST3_avg2"]
                    node[id "SST34_avg2" label "SST34_avg2"]
                    node[id "SST12_avg2" label "SST12_avg2"]
                    node[id "NATL_avg2" label "NATL_avg2"]
                    node[id "SATL_avg2" label "SATL_avg2"]
                    node[id "TROP_avg2" label "TROP_avg2"]
                    node[id "SOI_avg2" label "SOI_avg2"]
                    node[id "ESOI_avg2" label "ESOI_avg2"]
                    node[id "cpolr_avg2" label "cpolr_avg2"]
                    node[id "wpac850_avg2" label "wpac850_avg2"]
                    node[id "cpac850_avg2" label "cpac850_avg2"]
                    node[id "epac850_avg2" label "epac850_avg2"]
                    node[id "qbo_u30_avg2" label "qbo_u30_avg2"]
                    node[id "t2m_avg2" label "t2m_avg2"]
                    node[id "tp_avg2" label "tp_avg2"]
                    node[id "MPI" label "MPI"]
                    node[id "pop_density" label "pop_density"]
                    node[id "Year" label "Year"]
                    node[id "sin_month" label "sin_month"]
                    node[id "cos_month" label "cos_month"]
                    node[id "DANE_normalized" label "DANE_normalized"]
                    node[id "DANE_year_normalized" label "DANE_year_normalized"]
                    node[id "DANE_period_normalized" label "DANE_period_normalized"]
                    node[id "altitude" label "altitude"]

                    
                    edge[source "Year" target "SST4_avg2"]
                    edge[source "Year" target "excess"]
                    edge[source "Year" target "SST3_avg2"]
                    edge[source "Year" target "SST34_avg2"]
                    edge[source "Year" target "SST12_avg2"]
                    edge[source "Year" target "NATL_avg2"]
                    edge[source "Year" target "SATL_avg2"]
                    edge[source "Year" target "TROP_avg2"]
                    edge[source "Year" target "SOI_avg2"]
                    edge[source "Year" target "ESOI_avg2"]
                    edge[source "Year" target "cpolr_avg2"]
                    edge[source "Year" target "wpac850_avg2"]
                    edge[source "Year" target "cpac850_avg2"]
                    edge[source "Year" target "epac850_avg2"]
                    edge[source "Year" target "qbo_u30_avg2"]
                    edge[source "Year" target "t2m_avg2"]
                    edge[source "Year" target "tp_avg2"]
                    
        		    edge[source "sin_month" target "SST12_avg2"]
                    edge[source "sin_month" target "excess"]
                    edge[source "sin_month" target "SST3_avg2"]
                    edge[source "sin_month" target "SST34_avg2"]
                    edge[source "sin_month" target "SST4_avg2"]
                    edge[source "sin_month" target "NATL_avg2"]
                    edge[source "sin_month" target "SATL_avg2"]
                    edge[source "sin_month" target "TROP_avg2"]
                    edge[source "sin_month" target "SOI_avg2"]
                    edge[source "sin_month" target "ESOI_avg2"]
                    edge[source "sin_month" target "cpolr_avg2"]
                    edge[source "sin_month" target "wpac850_avg2"]
                    edge[source "sin_month" target "cpac850_avg2"]
                    edge[source "sin_month" target "epac850_avg2"]
                    edge[source "sin_month" target "qbo_u30_avg2"]
                    edge[source "sin_month" target "t2m_avg2"]
                    edge[source "sin_month" target "tp_avg2"]

                    edge[source "cos_month" target "SST12_avg2"]
                    edge[source "cos_month" target "excess"]
                    edge[source "cos_month" target "SST3_avg2"]
                    edge[source "cos_month" target "SST34_avg2"]
                    edge[source "cos_month" target "SST4_avg2"]
                    edge[source "cos_month" target "NATL_avg2"]
                    edge[source "cos_month" target "SATL_avg2"]
                    edge[source "cos_month" target "TROP_avg2"]
                    edge[source "cos_month" target "SOI_avg2"]
                    edge[source "cos_month" target "ESOI_avg2"]
                    edge[source "cos_month" target "cpolr_avg2"]
                    edge[source "cos_month" target "wpac850_avg2"]
                    edge[source "cos_month" target "cpac850_avg2"]
                    edge[source "cos_month" target "epac850_avg2"]
                    edge[source "cos_month" target "qbo_u30_avg2"]
                    edge[source "cos_month" target "t2m_avg2"]
                    edge[source "cos_month" target "tp_avg2"]
                    
                           
                    edge[source "SST3_avg2" target "SST34_avg2"]
                    edge[source "SST3_avg2" target "SST12_avg2"]
                    edge[source "SST3_avg2" target "NATL_avg2"]
                    edge[source "SST3_avg2" target "SATL_avg2"]
                    edge[source "SST3_avg2" target "TROP_avg2"]
                    edge[source "SST3_avg2" target "SOI_avg2"]
                    edge[source "SST3_avg2" target "ESOI_avg2"]
                    edge[source "SST3_avg2" target "cpolr_avg2"]
                    edge[source "SST3_avg2" target "wpac850_avg2"]
                    edge[source "SST3_avg2" target "cpac850_avg2"]
                    edge[source "SST3_avg2" target "epac850_avg2"]
                    edge[source "SST3_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST34_avg2" target "SST12_avg2"]
                    edge[source "SST34_avg2" target "NATL_avg2"]
                    edge[source "SST34_avg2" target "SATL_avg2"]
                    edge[source "SST34_avg2" target "TROP_avg2"]
                    edge[source "SST34_avg2" target "SOI_avg2"]
                    edge[source "SST34_avg2" target "ESOI_avg2"]
                    edge[source "SST34_avg2" target "cpolr_avg2"]
                    edge[source "SST34_avg2" target "wpac850_avg2"]
                    edge[source "SST34_avg2" target "cpac850_avg2"]
                    edge[source "SST34_avg2" target "epac850_avg2"]
                    edge[source "SST34_avg2" target "qbo_u30_avg2"]
                                       
                    edge[source "NATL_avg2" target "SATL_avg2"]
                    edge[source "NATL_avg2" target "TROP_avg2"]
                    edge[source "NATL_avg2" target "SOI_avg2"]
                    edge[source "NATL_avg2" target "ESOI_avg2"]
                    edge[source "NATL_avg2" target "cpolr_avg2"]
                    edge[source "NATL_avg2" target "wpac850_avg2"]
                    edge[source "NATL_avg2" target "cpac850_avg2"]
                    edge[source "NATL_avg2" target "epac850_avg2"]
                    edge[source "NATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SATL_avg2" target "TROP_avg2"]
                    edge[source "SATL_avg2" target "SOI_avg2"]
                    edge[source "SATL_avg2" target "ESOI_avg2"]
                    edge[source "SATL_avg2" target "cpolr_avg2"]
                    edge[source "SATL_avg2" target "wpac850_avg2"]
                    edge[source "SATL_avg2" target "cpac850_avg2"]
                    edge[source "SATL_avg2" target "epac850_avg2"]
                    edge[source "SATL_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "TROP_avg2" target "SOI_avg2"]
                    edge[source "TROP_avg2" target "ESOI_avg2"]
                    edge[source "TROP_avg2" target "cpolr_avg2"]
                    edge[source "TROP_avg2" target "wpac850_avg2"]
                    edge[source "TROP_avg2" target "cpac850_avg2"]
                    edge[source "TROP_avg2" target "epac850_avg2"]
                    edge[source "TROP_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SOI_avg2" target "ESOI_avg2"]
                    edge[source "SOI_avg2" target "cpolr_avg2"]
                    edge[source "SOI_avg2" target "wpac850_avg2"]
                    edge[source "SOI_avg2" target "cpac850_avg2"]
                    edge[source "SOI_avg2" target "epac850_avg2"]
                    edge[source "SOI_avg2" target "qbo_u30_avg2"]

                    edge[source "ESOI_avg2" target "cpolr_avg2"]
                    edge[source "ESOI_avg2" target "wpac850_avg2"]
                    edge[source "ESOI_avg2" target "cpac850_avg2"]
                    edge[source "ESOI_avg2" target "epac850_avg2"]
                    edge[source "ESOI_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpolr_avg2" target "wpac850_avg2"]
                    edge[source "cpolr_avg2" target "cpac850_avg2"]
                    edge[source "cpolr_avg2" target "epac850_avg2"]
                    edge[source "cpolr_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "wpac850_avg2" target "cpac850_avg2"]
                    edge[source "wpac850_avg2" target "epac850_avg2"]
                    edge[source "wpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "cpac850_avg2" target "epac850_avg2"]
                    edge[source "cpac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "epac850_avg2" target "qbo_u30_avg2"]
                    
                    edge[source "SST3_avg2" target "SST4_avg2"]
                    edge[source "SST3_avg2" target "excess"]
                    edge[source "SST34_avg2" target "SST4_avg2"]
                    edge[source "SST34_avg2" target "excess"]
                    edge[source "SST12_avg2" target "SST4_avg2"]
                    edge[source "SST12_avg2" target "excess"]
                    edge[source "NATL_avg2" target "SST4_avg2"]
                    edge[source "NATL_avg2" target "excess"]
                    edge[source "SATL_avg2" target "SST4_avg2"]
                    edge[source "SATL_avg2" target "excess"]
                    edge[source "TROP_avg2" target "SST4_avg2"]
                    edge[source "TROP_avg2" target "excess"]
                    edge[source "SOI_avg2" target "SST4_avg2"]
                    edge[source "SOI_avg2" target "excess"]
                    edge[source "ESOI_avg2" target "SST4_avg2"]
                    edge[source "ESOI_avg2" target "excess"]
                    edge[source "cpolr_avg2" target "SST4_avg2"]
                    edge[source "cpolr_avg2" target "excess"]
                    edge[source "wpac850_avg2" target "SST4_avg2"]
                    edge[source "wpac850_avg2" target "excess"]
                    edge[source "cpac850_avg2" target "SST4_avg2"]
                    edge[source "cpac850_avg2" target "excess"]
                    edge[source "epac850_avg2" target "SST4_avg2"]
                    edge[source "epac850_avg2" target "excess"]
                    edge[source "qbo_u30_avg2" target "SST4_avg2"]
                    edge[source "qbo_u30_avg2" target "excess"]
                    
                    edge[source "SST3_avg2" target "t2m_avg2"]
                    edge[source "SST3_avg2" target "tp_avg2"]
                    
                    edge[source "SST34_avg2" target "t2m_avg2"]
                    edge[source "SST34_avg2" target "tp_avg2"]
                    
                    edge[source "SST12_avg2" target "t2m_avg2"]
                    edge[source "SST12_avg2" target "tp_avg2"]
                    
                    edge[source "NATL_avg2" target "t2m_avg2"]
                    edge[source "NATL_avg2" target "tp_avg2"]
                    
                    edge[source "SATL_avg2" target "t2m_avg2"]
                    edge[source "SATL_avg2" target "tp_avg2"]
                    
                    edge[source "TROP_avg2" target "t2m_avg2"]
                    edge[source "TROP_avg2" target "tp_avg2"]
                    
                    edge[source "SOI_avg2" target "t2m_avg2"]
                    edge[source "SOI_avg2" target "tp_avg2"]

                    edge[source "ESOI_avg2" target "t2m_avg2"]
                    edge[source "ESOI_avg2" target "tp_avg2"]

                    edge[source "cpolr_avg2" target "t2m_avg2"]
                    edge[source "cpolr_avg2" target "tp_avg2"]
                    
                    edge[source "wpac850_avg2" target "t2m_avg2"]
                    edge[source "wpac850_avg2" target "tp_avg2"]
                    
                    edge[source "cpac850_avg2" target "t2m_avg2"]
                    edge[source "cpac850_avg2" target "tp_avg2"]
                    
                    edge[source "epac850_avg2" target "t2m_avg2"]
                    edge[source "epac850_avg2" target "tp_avg2"]
                    
                    edge[source "qbo_u30_avg2" target "t2m_avg2"]
                    edge[source "qbo_u30_avg2" target "tp_avg2"]
                    
                    edge[source "SST4_avg2" target "t2m_avg2"]
                    edge[source "SST4_avg2" target "tp_avg2"]
                    
                    edge[source "t2m_avg2" target "excess"]
                    edge[source "tp_avg2" target "excess"]           
                    
                    edge[source "t2m_avg2" target "MPI"]
                    edge[source "tp_avg2" target "MPI"]
                    
                    edge[source "t2m_avg2" target "pop_density"]
                    edge[source "tp_avg2" target "pop_density"]                   
                  
                    edge[source "MPI" target "excess"]
                    
                    edge[source "MPI" target "pop_density"]

                    edge[source "pop_density" target "excess"]
                    
                    edge[source "DANE_normalized" target "excess"]
                    edge[source "DANE_year_normalized" target "excess"]
                    edge[source "DANE_period_normalized" target "excess"]
                    
                    edge[source "SST4_avg2" target "excess"]
                    
                    edge[source "altitude" target "t2m_avg2"]
                    
                    
                    node[id "consensus" label "consensus"]
                    edge[source "consensus" target "t2m_avg2"]
                    edge[source "consensus" target "tp_avg2"]
                    edge[source "consensus" target "excess"]
                    edge[source "SOI_avg2" target "consensus"]
                    edge[source "SST4_avg2" target "consensus"]
                    edge[source "SST12_avg2" target "consensus"]
                    edge[source "SST34_avg2" target "consensus"]
                    edge[source "SST3_avg2" target "consensus"]
                    
                    ]"""
                    )

#%% 

# Identifying effects
identified_estimand_SST4_avg2 = model_SST4_avg2.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_SST4_avg2)

#%%

# Model with DoWhy
estimate_SST4_avg2 = model_SST4_avg2.estimate_effect(
    identified_estimand_SST4_avg2,
    effect_modifiers=['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST4_t1'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_outcome":True,
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)

print(estimate_SST4_avg2)

#%%

# ATE 
econml_estimator_SST4_avg2 = estimate_SST4_avg2.estimator.estimator
effect_modifier_cols = ['altitude', 'DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'SST4_t1']
X_data = data[effect_modifier_cols]  
X_data = X_data.dropna()

ate_SST4_avg2 = econml_estimator_SST4_avg2.effect(
    X=X_data 
).mean()  

print(ate_SST4_avg2)

# CI
ate_ci_SST4_avg2 = econml_estimator_SST4_avg2.effect_interval(
    X=X_data,
    alpha=0.05
)

ci_lower_SST4_avg2 = ate_ci_SST4_avg2[0].mean()
ci_upper_SST4_avg2 = ate_ci_SST4_avg2[1].mean()

print(ci_lower_SST4_avg2)
print(ci_upper_SST4_avg2)

data_ATE.at[3, 'ATE'] = ate_SST4_avg2
data_ATE.at[3, 'ci_lower'] = ci_lower_SST4_avg2
data_ATE.at[3, 'ci_upper'] = ci_upper_SST4_avg2
print(data_ATE)

#%%

effect_SST4_avg2 = econml_estimator_SST4_avg2.effect(
    X=X_data  
)  

# CI
effect_ci_SST4_avg2 = econml_estimator_SST4_avg2.effect_interval(
    X=X_data,
    alpha=0.05
)

# CATE alt
alt = data_SST4_avg2['altitude']  

# Grid for alt
min_alt = alt.min()
max_alt = alt.max()
delta = (max_alt - min_alt) / 100
alt_grid = np.arange(min_alt, max_alt + delta - 0.001, delta)

# Medias
DANE_encoded_mean = data_SST4_avg2['DANE_normalized'].mean()
DANE_year_encoded_mean = data_SST4_avg2['DANE_year_normalized'].mean()
DANE_period_encoded_mean = data_SST4_avg2['DANE_period_normalized'].mean()
SST4_t1_mean = data_SST4_avg2['SST4_t1'].mean()

# Matrix for prdiction
X_test_grid = np.column_stack([
    alt_grid,
    np.full_like(alt_grid, DANE_encoded_mean),
    np.full_like(alt_grid, DANE_year_encoded_mean),
    np.full_like(alt_grid, DANE_year_encoded_mean),
    np.full_like(alt_grid, SST4_t1_mean),
])

# Predicction effect
treatment_effect = econml_estimator_SST4_avg2.effect(X_test_grid)

#ci
hte_lower2_cons, hte_upper2_cons = econml_estimator_SST4_avg2.effect_interval(X_test_grid, alpha=0.05)
    
plot_data = pd.DataFrame({
    'alt': alt_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

cate_plot = (
    ggplot(plot_data)
    + aes(x='alt', y='treatment_effect')
    + geom_line(color='black', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Altitude (m)', y='Effect of SST4 on excess dengue cases', title='d')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(
        plot_title=element_text(hjust=0.5, size=12),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10)
    )
)

print(cate_plot)    

#%%

#with common cause
random_SST4_avg2 = model_SST4_avg2.refute_estimate(identified_estimand_SST4_avg2, estimate_SST4_avg2,
                                         method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_SST4_avg2 )

#with subset
subset_SST4_avg2  = model_SST4_avg2.refute_estimate(identified_estimand_SST4_avg2, estimate_SST4_avg2,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_SST4_avg2) 
      
#with placebo 
placebo_SST4_avg2  = model_SST4_avg2.refute_estimate(identified_estimand_SST4_avg2, estimate_SST4_avg2,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_SST4_avg2)

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

labels = [
    'EL Niño region 1-2 (sd = 2.33 °C)',
    'EL Niño region 3 (sd = 1.24 °C)',
    'EL Niño region 34 (sd = 0.97 °C)',
    'EL Niño region4 (sd = 0.67 °C)'
]

colors = ['red', 'green', 'orange', 'blue']

y_positions = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

for i, (ate, ci_lower, ci_upper) in enumerate(data_ATE.values):
    ax.plot([ci_lower, ci_upper], [y_positions[i], y_positions[i]], 
            color=colors[i], linewidth=2.5)
    
    ax.plot(ate, y_positions[i], marker='s', markersize=8, 
            color=colors[i], markerfacecolor=colors[i], markeredgecolor='black', 
            markeredgewidth=1)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels)

ax.set_xlabel('Average Treatment Effect (ATE)')
ax.set_xlim(-0.05, 0.1)  
ax.grid(True, alpha=0.3)

ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)

ax.set_title('', fontsize=14, pad=20)

plt.tight_layout()

plt.show()

#%%