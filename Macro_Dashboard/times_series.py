import numpy as np
import pandas as pd
import cmath as math
from arch import arch_model
from datetime import datetime



def dev_mean(series):
    import pandas as pd
    import numpy as np
    """
    Returns 1-month, 3-month and long-term mean and standard deviation of imputed series
    """
    series=series.interpolate(method='linear')
    #1 month
    series_mean_1m=series.rolling(window=30).mean()
    series_std_1m=series.rolling(window=30).std()
    series_dev_1mupper=series_mean_1m+series_std_1m
    series_dev_1mlower=series_mean_1m-series_std_1m
    
    #3 month
    series_mean_3m=series.rolling(window=90).mean()
    series_std_3m=series.rolling(window=90).std()
    series_dev_3mupper=series_mean_3m+series_std_3m
    series_dev_3mlower=series_mean_3m-series_std_3m

    #long-term
    series_mean_lt=pd.Series(series.mean(),index=series.index)
    series_std_lt=pd.Series(series.std(),index=series.index)
    series_dev_ltupper=series_mean_lt+series_std_lt
    series_dev_ltlower=series_mean_lt-series_std_lt
    
    dev_mean_data=pd.DataFrame(series_mean_1m)
    dev_mean_data[1]=series_dev_1mupper
    dev_mean_data[2]=series_dev_1mlower
    dev_mean_data[3]=series_mean_1m
    dev_mean_data[4]=series_dev_3mupper
    dev_mean_data[5]=series_dev_3mlower
    dev_mean_data[6]=series_mean_lt
    dev_mean_data[7]=series_dev_ltupper
    dev_mean_data[8]=series_dev_ltlower
    dev_mean_data.columns = ['1mavg', '1mupper', '1mlower', '3mavg', '3mupper', '3mlower', 'ltavg', 'ltupper', 'ltlower'] 
    
    prefix=str(series.name)
    dev_mean_data.add_prefix(prefix)
    
    return dev_mean_data

def dev_mean_table(series):
    import pandas as pd
    import numpy as np
    """
    Returns 1-month, 3-month and long-term deviation from mean measured in std
    """
    series=series.interpolate(method='linear')
    #1 month
    series_mean_1m=series.rolling(window=30).mean()
    series_std_1m=series.rolling(window=30).std()
    series_dev_1m=(series-series_mean_1m)/(series_std_1m)
    
    #3 month
    series_mean_3m=series.rolling(window=90).mean()
    series_std_3m=series.rolling(window=90).std()
    series_dev_3m=(series-series_mean_3m)/(series_std_3m)


    #long-term
    series_mean_lt=pd.Series(series.mean(),index=series.index)
    series_std_lt=pd.Series(series.std(),index=series.index)
    series_dev_lt=(series-series_mean_lt)/(series_std_lt)

    
    series_dev=pd.DataFrame(series_dev_1m)
    series_dev[1]=series_dev_3m
    series_dev[2]=series_dev_lt
    series_dev.columns = ['1-Month', '3-Month', 'Long-Term']
                             
    prefix=str(series.name)
    series_dev.add_prefix(prefix)
    
    return series_dev

def highlight_cells_dev(val):
    color1='indianred'
    color2='lightcoral'
    color3='lightgreen'
    color4='gren'
    if val<=-1:
        return 'background-color: {}'.format(color1)
    elif val>-1 and val<0:
        return 'background-color: {}'.format(color2)
    elif val<1 and val>0:
        return 'background-color: {}'.format(color3)
    elif val>=1:
        return 'background-color: {}'.format(color4)

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected. Test

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    maxlag=12
    test = 'ssr_chi2test'
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    df_1 = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            lag = np.argmin(p_values) + 1
            df.loc[r, c] = min_p_value
            df_1.loc[r, c] = lag
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df,df_1

# Function to perform ADF test and display results
def adf_test_for_column(column):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(column)
    print(f"ADF Statistic for {column.name}: {result[0]}")
    print(f"P-value for {column.name}: {result[1]}")
    print("Critical Values:", result[4])

    if result[1] <= 0.05:
        print(f"Reject the null hypothesis for {column.name}. The data is stationary.")
    else:
        print(f"Fail to reject the null hypothesis for {column.name}. The data is non-stationary.")


def params_garch(data, max_p, max_q):
    import itertools
    """
    Returns the optimal parameters, GARCH model and results 
    """
    best_aic = np.inf
    best_bic = np.inf
    best_p = 0
    best_q = 0
    
    for p, q in itertools.product(range(max_p+1), range(max_q+1)):
        if p == 0 and q == 0:
            continue
            
        try:
            model = arch.arch_model(data, vol='Garch', p=p, q=q)
            result = model.fit(disp='off')
            aic = result.aic
            bic = result.bic
            
            if aic < best_aic and result.pvalues[-1] < 0.05:
                best_aic = aic
                best_p = p
                best_q = q
            
            if bic < best_bic and result.pvalues[-1] < 0.05:
                best_bic = bic
                best_p = p
                best_q = q
                
        except:
            continue
    
    return best_p, best_q

# Example usage:
# Assuming 'data' is your time series data
# Replace it with your actual data
# best_params, best_aic, best_bic = find_smallest_garch(data)    


def params_arch(data, max_p):
    import itertools
    """
    Returns the optimal parameters, ARCH model and results 
    """
    best_aic = np.inf
    best_bic = np.inf
    best_p = 0
        
    for p in range(max_p+1):
        if p == 0:
           continue
            
        try:
            model = arch.arch_model(data, vol='ARCH', p=p)
            result = model.fit(disp='off')
            aic = result.aic
            bic = result.bic
            
            if aic < best_aic and result.pvalues[-1] < 0.05:
                best_aic = aic
                best_p = p
            
            if bic < best_bic and result.pvalues[-1] < 0.05:
                best_bic = bic
                best_p = p
                
        except:
            continue
    
    return best_p

# Example usage:
# Assuming you have a pandas DataFrame 'returns' containing your return data
# best_p, best_aic, best_bic = find_best_arch_params(returns, max_p=5)


def data_mom(data, type):
    
    if type == 'pct':
        def mom(x):
            mom = (x - x.shift(1))
            annmom = mom*12
            ann3m = (x-x.shift(3))*4
            ann6m = (x-x.shift(6))*2
            yoy = (x-x.shift(12))
            return mom, annmom, ann3m, ann6m, yoy
        
    else:
        def mom(x):
            mom = 100*(x).pct_change()
            annmom = 100*(x).pct_change()*12
            ann3m = 100*(((x/x.shift(3))-1)*4)
            ann6m = 100*(((x/x.shift(6))-1)*2)
            yoy = 100*((x/x.shift(12))-1)
            return mom, annmom, ann3m, ann6m, yoy
        
    for i in data.columns:
        col_names = [f'{i}_mom', f'{i}_annmom', f'{i}_ann3m', f'{i}_ann6m', f'{i}_yoy']
        data_a = pd.DataFrame(mom(data[i]))
        data_a.index = col_names
        data_b = data_a.T
        data[col_names] = data_b

    data = data.dropna()

def year_month(data):
    data.index = pd.to_datetime(data.index).strftime('%Y-%m')
    return data
