import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


def product_time_series_stationarity( product, col_names ):
    ## Data is stationary?
    ## Use Augmented Dickey-Fuller test
    ## onle check p-value 
    ## if p-value < 0.05 it stationary, if its greater its non stationary
    # col_names = ["sales", "py_sales", "prices", "py_prices"]
    names = ["col_names", "adf_stat", "p_value", "is_stationary"]
    data_dict = {}
    data_lists = [col_names, [],[],[]]
    # adf_stat = []
    # p_val = []
    # is_stationary = []
    for col in col_names:
        ad_fuller_res = adfuller(product[col])
        # adf_stat.append(ad_fuller_res[0])
        # p_val.append(ad_fuller_res[1])
        # is_stationary.append(ad_fuller_res[1] < 0.05)
        data_lists[1].append(ad_fuller_res[0])
        data_lists[2].append(ad_fuller_res[1])
        data_lists[3].append(ad_fuller_res[1] < 0.05)
    for i in range(len(names)):
        data_dict[names[i]] = data_lists[i]
    return pd.DataFrame(data_dict)


def x_causes_y( x, y, data, N_lags=4, info=False ):
    ## Time series are correlated??
    ## granger causality test check if the second time series causes the first
    ## data must be stationary
    ## if p-value in k-lag is less than 0.05 then the hypothesis is true
    lags_corr = []
    lag_failure = []

    granger = grangercausalitytests(data[[y,x]], N_lags, verbose=info)
    
    if info:
        print(granger)
    
    for lag in granger:
        # print(lag)
        lag_corr = True
        dict_tests = granger[lag][0]
        for test_name in dict_tests:
            lag_corr = lag_corr and dict_tests[test_name][1] < 0.05
        lags_corr.append(lag_corr)
        if not lag_corr:
            lag_failure.append(lag)

    return False not in lags_corr


def causality_matrix( product, labels ):
    n = len(labels)
    aux = np.array([["-"*50]*n]*n)
    for i in range(n):
        for j in range(n):
            if i == j:
                aux[i,j] = "NULL"
            else:
                aux[i,j] = f"{labels[i]} causes {labels[j]}" if x_causes_y(labels[i], labels[j], product) else "No corr"
    df = pd.DataFrame(aux, columns=labels)
    df2 = pd.DataFrame({"names":labels})
    df = pd.concat([df2,df], axis=1)

    return df