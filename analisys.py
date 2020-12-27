# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %%
# get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')
# %%pycodestyle
# https://stackoverflow.com/a/54278757

# export venv to file
# https://stackoverflow.com/a/14685017

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_regression
from joblib import dump, load


weo_subject_code = 'WEO Subject Code'
estimates_after = 'Estimates Start After'
iso_col = 'ISO'
country_name = 'Country'
usecols = [iso_col, weo_subject_code, country_name]
for i in range(1980, 2020):
    usecols.append(str(i))
usecols.append(estimates_after)

df = pd.read_table(
    'WEOOct2020all.xls',
    encoding='UTF-16-LE',
    usecols=usecols,
)


# %%
''' select top 10 GDP per capita countries '''

# filtering gdp per capita
gdppc = 'NGDPRPPPPC'
def gdp_per_capita_common_dollar(col): return col[weo_subject_code] == gdppc


gdp_per_capita_df = df.loc[gdp_per_capita_common_dollar]

# https://stackoverflow.com/a/52065957
# gdp_per_capita_df
#  = gdp_per_capita_df['2019'].astype('str').str.replace(',', '')

# creating dataframe for sorting
# https://stackoverflow.com/a/57064872
year_col = '2019'
year_col_before = str(int(year_col)-1)
gdp_increase_col = f'GDP Increase from {year_col_before} to {year_col}'

# requires converting to numeric value for sorting
gdp_per_capita_df[year_col] = gdp_per_capita_df[year_col].replace(
    regex=',', value='').astype(float)
gdp_per_capita_df[year_col_before] = gdp_per_capita_df[
    year_col_before].replace(regex=',', value='').astype(float)
gdp_per_capita_df = gdp_per_capita_df[[
    country_name, year_col, year_col_before]]
gdp_per_capita_df = gdp_per_capita_df.set_index(country_name)

# calculate difference between two columns row by row
# https://towardsdatascience.com/time-series-modeling-using-scikit-pandas-and-numpy-682e3b8db8d1
# gdp_per_capita_df.loc[:,gdp_increase_col]
#  = gdp_per_capita_df.loc[:, year_col_before].diff()
# gdp_per_capita_df[gdp_increase_col]
#  = gdp_per_capita_df[year_col] - gdp_per_capita_df[year_col]

# ran into issues that calculation showed zeros every where
# lost datatype :/
c = []
for row in gdp_per_capita_df.itertuples():
    # c.append(
    # [row[Index],
    # row[year_col],
    # row[year_col_before],
    # row[year_col] - row[year_col_before]])
    c.append([row[0], row[1], row[2], row[1] - row[2]])

delta_col = np.array(c)
delta_df = pd.DataFrame(
    delta_col,
    columns=[country_name, year_col, year_col_before, gdp_increase_col])
delta_df = delta_df.set_index(country_name).astype(float)
delta_df = delta_df.dropna()

# sorting on gdp increase col
delta_df = delta_df.sort_values(
    gdp_increase_col, ascending=False, na_position='last')

# select top 10
delta_df[:10]

# select first row of column
# https://stackoverflow.com/a/25254087
# gdp_per_capita_df.iloc[0, gdp_per_capita_df.columns.get_loc(year_col)]

# https://stackoverflow.com/a/64307654
# df[gdp_per_capita_common_dollar].values
# for i, row in enumerate(df[gdp_per_capita_common_dollar].values):
#     if row[2] < 1:
#         print(i,row)

# df_only_iso =  df[gdp_per_capita_common_dollar][iso_col]
# df_only_iso


# %%

oecd_countries_all_caps = {
    'AUSTRIA': '',
    'AUSTRALIA': '',
    'BELGIUM': '',
    'CANADA': '',
    'CHILE': '',
    'COLOMBIA': '',
    'CZECH REPUBLIC': '',
    'DENMARK': '',
    'ESTONIA': '',
    'FINLAND': '',
    'FRANCE': '',
    'GERMANY': '',
    'GREECE': '',
    'HUNGARY': '',
    'ICELAND': '',
    'IRELAND': '',
    'ISRAEL': '',
    'ITALY': '',
    'JAPAN': '',
    'KOREA': '',
    'LATVIA': '',
    'LITHUANIA': '',
    'LUXEMBOURG': '',
    'MEXICO': '',
    'NETHERLANDS': '',
    'NEW ZEALAND': '',
    'NORWAY': '',
    'POLAND': '',
    'PORTUGAL': '',
    'SLOVAK REPUBLIC': '',
    'SLOVENIA': '',
    'SPAIN': '',
    'SWEDEN': '',
    'SWITZERLAND': '',
    'TURKEY': '',
    'UNITED KINGDOM': '',
    'UNITED STATES': ''}
# properly formatting OECD country names
oecd_countries = {}
for key in oecd_countries_all_caps:
    oecd_countries[key.title()] = ''
# oecd_countries


# %%

'''
plot of OECD countries population
'''

population_key = 'LP'
# selecting dataframe based on two columns
# population_df = df.loc[
#  (df[weo_subject_code] == population_key) & (df[country_name] == 'Lithuania')
# ]

# creating dataframe on population
population_df = df.loc[(df[weo_subject_code] == population_key)]

# sets index and index is stored for future
population_df = population_df.set_index(country_name)

# filtering oecd contries
population_df = population_df.loc[oecd_countries]

# do not convert to string for filtering
# rather convert dataframe to floats
decade = []
for i in range(2010, 2020):
    decade.append(str(i))

# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
# selecting decade colums
population_df = population_df[decade]
# converting to plotable time series, transpose
population_df = population_df.T

# wrong data type is set needs to cast to numeris type
population_df = population_df.astype(float)

# select only limited subset of countries instead of all as it population
# change is better visible
population_df = population_df[['Lithuania', 'Latvia', 'Iceland']]


plt.figure()
population_df.plot()


# %%
'''save all countries GDP in different PNG files'''

# Gross domestic product, current prices
# Values are based upon GDP in national currency converted to U.S. dollars
gdp_key = 'NGDPD'

gdp_df = df.loc[lambda df: df[weo_subject_code] == gdp_key]

# setting index country name
gdp_df = gdp_df.set_index(country_name)

# filtering non required columns
selected_cols = []

for col in gdp_df.columns:
    if col == iso_col or col == weo_subject_code or col == estimates_after:
        continue

    selected_cols.append(col)

# select country name column and period columns
gdp_df = gdp_df[selected_cols]

# https://stackoverflow.com/a/49896522
# applies lambda to rows, cleans numeric values of thousands separator
gdp_df = gdp_df.apply(lambda df: df.str.replace(',', '').astype(float), axis=0)

# prepare folder for pics
figure_folder_name = 'figures'
if not os.path.exists(figure_folder_name):
    os.makedirs(figure_folder_name)

# remove [0:1] to save all
for country in gdp_df.index[0:1]:
    # print(country)
    # https://stackoverflow.com/a/45379210
    fig = gdp_df.loc[country].plot().get_figure()
    # https://stackoverflow.com/a/4805178
    # if face color is not set explicitly its trasnparet, wtf
    fig.savefig(f'{figure_folder_name}\{country}.png',
                format='png', transparent=False, facecolor='white')
    # closes the plot, as no need to display, while saving
    # https://stackoverflow.com/a/15713545
    plt.close(fig)


# %%
''' find lowest common denominator for year 2015 '''
# create dataframe of WEO codes and 2015 year
common_denominator_df = df[[weo_subject_code, '2015']]

# removes all na values
common_denominator_df = common_denominator_df.dropna()

# selecting only WEO code and grouping to display
common_denominator_df = common_denominator_df[[weo_subject_code]]
common_denominator_df = common_denominator_df.groupby([weo_subject_code])

# common_denominator_df.apply(print)
# https://stackoverflow.com/a/36951842
# simple print
lowest_common_denom = []
for key in common_denominator_df.groups.keys():
    lowest_common_denom.append(key)

# uncoment for display
# lowest_common_denom


# %%
''' K-Means clustering '''

# still requires GDP_key
volume_of_exported_goods_key = 'TXG_RPCH'


kmeans = KMeans(n_clusters=5)


# %%
''' GDP per capita prediction '''

# select non GDP related weo keys
gdp_weo_key = 'GDP'


def country_select(df): return df[country_name] == 'Germany'


# use regular expr with bitmask to filter out all GDP related fields
# https://stackoverflow.com/a/17097777
input_features = df[
    ~df[weo_subject_code].str.contains(gdp_weo_key, na=False)
    & ~df[weo_subject_code].str.contains('PPP', na=False)].fillna(0.0)


def gdp_per_capita_common_dollar(col): return col[weo_subject_code] == gdppc


gdppc = 'NGDPRPPPPC'

# drop_clumns = [country_name,iso_col, weo_subject_code, estimates_after]


def prepare_country_data(input_features, country_select):
    drop_clumns = [country_name, iso_col, estimates_after]

    input_features = input_features.loc[country_select]
    # input_features = input_features.set_index(country_name)
    input_features = input_features.drop(columns=drop_clumns)
    input_features = input_features.T

    # save feature column codes
    feature_weo_codes = []

    # building feature data frame
    cleand_features = []
    i = 0
    for row in input_features.itertuples():
        if len(row) < 21:
            continue
        # index
        if row[0] == weo_subject_code:
            feature_weo_codes = row[1:]
            continue

        float_fts = []

        for x in row[1:]:

            if type(x) == str:
                if x == '--':
                    float_fts.append(0.0)
                else:
                    float_fts.append(float(x.replace(',', '')))
            else:
                float_fts.append(float(x))
        cleand_features.append(float_fts)

    # cleand_features
    # print(feature_weo_codes)

    drop_columns_inlcuding_weo = [country_name,
                                  iso_col, estimates_after, weo_subject_code]

    gdp_data = df.loc[gdp_per_capita_common_dollar].dropna()
    gdp_data = gdp_data.loc[country_select].drop(
        columns=drop_columns_inlcuding_weo)
    # gdp_data = gdp_data.apply(
    # lambda df: df.str.replace(',','').astype(float), axis=0)
    cleand_result = []
    for row in gdp_data.T.itertuples():
        if len(row) < 2:
            continue

        cleand_result.append(float(row[1].replace(',', '')))

    return (cleand_features, cleand_result, feature_weo_codes)


def select_training_countries():
    countries_list = input_features[[country_name]]
    countries_list = countries_list.set_index(country_name)
    countries_list = countries_list.groupby(country_name)
    filtered_countries = []
    for i, key in enumerate(countries_list.groups.keys()):
        if type(key) != str:
            continue

        if i % 3 == 0 or key == 'Germany' or key == 'Italy':
            filtered_countries.append(key)

    return filtered_countries


failed_result_gen = []
cleand_features = []
cleand_result = []
feature_weo_codes = []
for c_name in select_training_countries():
    gdp_features, gdp_result, weo_codes = prepare_country_data(
        input_features=input_features,
        country_select=lambda df: df[country_name] == c_name)
    # print(c_name, cleand_features, cleand_result)
    if len(gdp_result) < 1:
        failed_result_gen.append(c_name)
        continue

    feature_weo_codes = weo_codes
    # append features and results
    for row in gdp_features:
        cleand_features.append(row)
        # print(cleand_features)

    for x in gdp_result:
        cleand_result.append(x)

print(np.array(cleand_features).shape, np.array(cleand_result).shape)
# print(
# failed_result_gen, len(failed_result_gen), len(select_training_countries()))

# cleand_result

slice_length = -600
training_features = cleand_features[:slice_length]
training_gdp = cleand_result[:slice_length]

training_features_test = cleand_features[slice_length:]
training_gdp_test = cleand_result[slice_length:]

# training
# https://machinelearningmastery.com/make-predictions-scikit-learn/
# https://machinelearningmastery.com/calculate-feature-importance-with-python/

model = LinearRegression()
model.fit(training_features, training_gdp)


def get_filter_coeficients_for_manual_selection():
    filter_coefs = []
    # # get importance
    importance = model.coef_
    # summarize feature importance
    for i, v in enumerate(importance):
        # if v > 100:
        print_t = (i, v, feature_weo_codes[i])
        print('Feature: %0d, Score: %.5f, weo_key: %s' % print_t)
        filter_coefs.append(print_t)
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

    return filter_coefs

# print(get_filter_coeficients_for_manual_selection())


# prediction
predict = training_features_test
result = training_gdp_test

result_predicted = model.predict(predict)

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(result, result_predicted))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(result, result_predicted))

# print('real result: ', result, result_predicted[0])

# # https://scikit-learn.org/stable/modules/model_persistence.html
# file_name = 'filename.joblib'
# dump(model, file_name)
# model2 = load(file_name)

# result_predicted2 = model2.predict(predict)
# print('Coefficients: \n', model2.coef_
#       % mean_squared_error(result, result_predicted2))
# print('Coefficient of determination: %.2f'
#       % r2_score(result, result_predicted2))


# %%
''' auto feature selection '''
# https://machinelearningmastery.com/feature-selection-for-regression-data/
# feature selection


def select_features(X_train, y_train, X_test):
    # configure to select all features
    # fs = SelectKBest(score_func=f_regression, k='all')
    fs = SelectKBest(score_func=mutual_info_regression, k=5)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs, X_test_fs, fs = select_features(
    training_features, training_gdp, training_features_test)

# get selected features
# https://stackoverflow.com/a/43765224

auto_selected_features = fs.get_support(indices=True)
print(auto_selected_features)
print(feature_weo_codes)
best_features = []
for i in auto_selected_features:
    best_features.append(feature_weo_codes[i])
print(best_features)

# fit the model
model5 = LinearRegression()
model5.fit(X_train_fs, training_gdp)
# evaluate the model
yhat = model5.predict(X_test_fs)
# evaluate predictions
mse = mean_squared_error(training_gdp_test, yhat)
print('MSE: %.2f' % mse)

file_name = 'filename5.joblib'
dump(model5, file_name)


# # shows coefs and plot
# print(X_train_fs.shape)
# # what are scores for the features
# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
# plt.show()
