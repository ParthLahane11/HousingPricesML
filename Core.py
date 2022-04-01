import Fetch
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_housing_data(housing_path = Fetch.HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

def split_data():
    train_set, test_set = train_test_split(housings, test_size= 0.2, random_state = 42)

def strat_sampling():
    housings['income_cat'] = pd.cut(housings['median_income'], bins = [0., 1.5, 3.0, 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])

def strat_split():
    split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state= 42)
    for train_index, test_index in split.split(housings, housings['income_cat']):
        strat_train_set = housings.loc[train_index]
        strat_test_set  = housings.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis = 1, inplace = True)
    housing = strat_train_set.drop('median_house_value', axis = 1)
    housing_label = strat_train_set['median_house_value'].copy()
    return housing, housing_label
    # housings.plot(kind ='scatter', x = 'longitude', y = 'latitude', alpha = 0.4, s = housings['population']/100, label = 'population', figsize = (10, 7), c = 'median_house_value', cmap = plt.get_cmap('jet'), colorbar = True,)
    # plt.legend()
    # plt.show()
    # housings['rooms_per_household'] = housings['total_rooms']/housings['households']
    # housings['bedrooms_per_room'] = housings['total_bedrooms']/housings['total_rooms']
    # housings['population_per_household'] = housings['population']/housings['households']
    # corr_matrix = housings.corr()
    # disp = corr_matrix['median_house_value'].sort_values(ascending=False)
    # print(disp)
    # attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    # scatter_matrix(housings[attributes], figsize=(12, 8))
    # plt.show()
def fix_missing_values(Features):
    imputer = SimpleImputer(strategy = 'median')
    housing_num = housings.drop('ocean_proximity', axis = 1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns = housing_num.coulmns, index = housing_num.index)
    return housing_tr

def encode_text_columns(Features):
    O_Hot = OneHotEncoder()
    Features_Text = Features[['ocean_proximity']]
    Features_Text = O_Hot.fit_transform(Features_Text)
    Features_Text = pd.DataFrame(Features_Text)
    return Features_Text

Fetch.fetch_housing_data()
housings = load_housing_data()
strat_sampling()
Features, Labels = strat_split()
encode_text_columns(Features)
