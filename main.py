import json
from math import sqrt

from numpy import mean, std
from sklearn.model_selection import train_test_split, RepeatedKFold
import pandas as pd
from sklearn import neighbors, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


def trynorm(x):
    as_dict = json.loads(x)
    max_plan = max(as_dict, key=lambda x: x['billingEntity']['monthlyPrice'])
    return max_plan


app_id_name = {'1380b703-ce81-ff05-f115-39571d94dfcd': 'WixStores',
               '14bcded7-0066-7c35-14d7-466cb3f09103': 'WixBlog',
               '13d21c63-b5ec-5912-8397-c3a5ddb27a97': 'WixBookings',
               '1475ab65-206b-d79a-856d-fa10bdb479ea': 'WixReservations',
               '1522827f-c56c-a5c9-2ac9-00f9e6ae12d3': 'WixPricingPlans',
               '140603ad-af8d-84a5-2c80-a0f60cb47351': 'WixEvents',
               '13e8d036-5516-6104-b456-c8466db39542': 'WixRestaurantsOrders',
               '135aad86-9125-6074-7346-29dc6a3c9bcf': 'WixHotels',
               '147ab90e-91c5-21b2-d6ca-444c28c8a23b': 'WixArtStore'}


def extract_app_req(listing):
    load_json = json.loads(listing)[0]
    if 'info' in load_json:
        info = load_json['info']
        if 'listingInfoEntityV2' in info:
            listingInfoEntityV2 = info['listingInfoEntityV2']
            if 'installationRequirement' in listingInfoEntityV2:
                installationRequirement = listingInfoEntityV2['installationRequirement']
                if 'requiredApps' in installationRequirement:
                    requiredApps = installationRequirement['requiredApps']
                    return [app_id_name[requiredApps[i]["slug"]] for i in range(len(requiredApps))]


def extract_description_length(listing):
    load_json = json.loads(listing)[0]
    if 'info' in load_json:
        info = load_json['info']
        if 'listingInfoEntityV2' in info:
            listingInfoEntityV2 = info['listingInfoEntityV2']
            if 'basicInfo' in listingInfoEntityV2:
                basicInfo = listingInfoEntityV2['basicInfo']
                longDescription = str(basicInfo['longDescription']).split()
                return len(longDescription)


def extract_components(components):
    load_json = json.loads(components)[0]
    if 'compData' in load_json:
        if len(load_json['compData']) > 0:
            return list(load_json['compData'])[0]


def create_df():
    df = pd.read_csv("new.csv")
    # add maximum plan columns
    plans = df['plans'].tolist()
    plans_max = list(map(trynorm, plans))
    plans_pd = pd.DataFrame(plans_max)
    new_df = df.join(plans_pd, lsuffix="Pricing")

    # normalize billing entity
    billing = new_df['billingEntity']
    billing_norm = pd.json_normalize(billing)
    new_df = new_df.join(billing_norm, lsuffix="Billing")

    # add app requirements
    listing_info = new_df['listing_info'].tolist()
    app_reqs = list(map(extract_app_req, listing_info))
    app_reqs = pd.DataFrame(app_reqs)
    app_reqs.columns = ['appRequirements']
    new_df = new_df.join(app_reqs)

    # extract description length
    description_length = list(map(extract_description_length, listing_info))
    description_length = pd.DataFrame(description_length)
    description_length.columns = ['description_length']
    new_df = new_df.join(description_length)

    # count components
    component_list = new_df['components'].to_list()
    component = list(map(extract_components, component_list))
    component = pd.DataFrame(component)
    component.columns = ['component']
    new_df = new_df.join(component)
    dummies = pd.get_dummies(new_df['component'])
    new_df = new_df.join(dummies)

    new_df.to_csv("new_df_normalized.csv")

    # convert datetime created to features
    new_df['date_created'] = pd.to_datetime(new_df['date_created'])
    new_df['day_created'] = new_df['date_created'].dt.day
    new_df['month_created'] = new_df['date_created'].dt.month
    new_df['year_created'] = new_df['date_created'].dt.year
    new_df['hour_created'] = new_df['date_created'].dt.hour

    filtered_df = new_df[["app_id", "namePricing", "name", "discountPercent", "day_created", "month_created",
                          "year_created", "hour_created", "monthlyPrice", "yearlyPrice", "free_trial_days",
                          "is_required_wix_premium",  "is_wix_app", "appRequirements", "description_length",
                          'widgetComponentData', 'pageComponentData',
                          'embeddedScriptComponentData', 'dashboardComponentData']].copy()
    language_count = new_df['app_id'].value_counts().reset_index()
    language_count.columns = ['app_id', 'availableLanguagesCount']
    language_count.set_index('app_id')
    filtered_no_duplicates = filtered_df.drop_duplicates(subset=['app_id']).set_index('app_id')
    filtered_no_duplicates.index.name = 'app_id'

    test_apps = filtered_no_duplicates[filtered_no_duplicates['namePricing'].str.contains('\$\$\$')]
    filtered_no_duplicates = filtered_no_duplicates.append(test_apps)
    filtered_no_duplicates = filtered_no_duplicates.drop_duplicates(keep=False, subset=["namePricing"])

    filtered_with_language_count = pd.merge(filtered_no_duplicates, language_count, on='app_id')

    # deal with categorical column
    categorical = filtered_with_language_count['appRequirements'].apply(lambda d: d if isinstance(d, list) else [])
    dummies = pd.get_dummies(categorical.apply(pd.Series).stack()).sum(level=0)

    df = filtered_with_language_count.join(dummies).fillna(0.0)
    df.drop(['appRequirements'], axis=1, inplace=True)
    df.to_csv("new_df.csv")

    return df


def predict_monthly(df):
    df = df.reset_index(drop=True)
    df.drop(['app_id', 'namePricing', 'name', 'yearlyPrice', 'discountPercent'], axis=1, inplace=True)
    # df = df.iloc[:, 1:]

    df.drop(['WixHotels', 'WixPricingPlans', 'WixReservations', 'WixArtStore'], axis=1, inplace=True)

    # plt.figure(figsize=(12, 12))
    # cor = df.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()

    X = df.loc[:, df.columns != 'monthlyPrice']
    y = df['monthlyPrice']

    train, test = train_test_split(df, test_size=0.3, shuffle=True)

    x_train = train.drop('monthlyPrice', axis=1)
    y_train = train['monthlyPrice']

    x_test = test.drop('monthlyPrice', axis=1)
    y_test = test['monthlyPrice']


    # rmse_val = []  # to store rmse values for different k
    # for K in range(20):
    #     K = K + 1
    #     model = neighbors.KNeighborsRegressor(n_neighbors=K)
    #
    #     model.fit(x_train, y_train)  # fit the model
    #     pred = model.predict(x_test)  # make prediction on test set
    #     error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
    #     rmse_val.append(error)  # store rmse values
    #     print('RMSE value for k= ', K, 'is:', error)

    cv = RepeatedKFold(n_splits=3, n_repeats=10, random_state=2)
    scoring = 'neg_mean_absolute_error'
    print("\nKNN")
    regression_model = make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor(n_neighbors=12))
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "LinearRegression")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')


    print("\nLinear Regression")
    regression_model = make_pipeline(StandardScaler(), linear_model.LinearRegression())
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "LinearRegression")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')


    print("\nKernelRidge")
    regression_model = make_pipeline(StandardScaler(), KernelRidge())
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "KernelRidge")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')


    print("\nElastic Net")
    regression_model = make_pipeline(StandardScaler(), ElasticNet())
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "ElasticNet")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')


    print("\nGradientBoostingRegressor")
    regression_model = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "GradientBoostingRegressor")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')


    print("\nSVR")
    regression_model = make_pipeline(StandardScaler(), SVR())
    predict_with_model(regression_model, x_train, y_train, x_test, y_test, "SVR")
    cross_score = cross_val_score(regression_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    print(f"Cross val score: {cross_score}")
    print(f'Accuracy: {mean(cross_score)}, ({std(cross_score)})')



def predict_with_model(regression_model, x_train, y_train, x_test, y_test, name):
    regression_model.fit(x_train, y_train)
    pred = regression_model.predict(x_test)
    copy_df = x_test.copy(deep=True)
    copy_df['predMonthlyPrice'] = pred
    copy_df['actualMonthlyPrice'] = y_test
    copy_df.to_csv(f"{name}.csv")
    print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
    print("Coefficient of determination: %.2f" % mean_absolute_error(y_test, pred))


if __name__ == '__main__':
    df = create_df()
    # df = pd.read_csv('df.csv')
    predict_monthly(df)

