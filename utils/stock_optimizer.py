
# import packages
import pandas as pd 
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

def prep_data(df, tickers):
    '''
    prep_data:
    - Parameters:
        + df: DataFrame
        + tickers: list of tickers for optimization
    - Output: Return data ready for optimization model
    '''
    df_filtered = df.loc[(df['Ticker'].isin(tickers)) & (df['Type'] == 'Close')]
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    pivot_df = df_filtered.pivot(index = 'Date', columns = 'Ticker', values = 'Price').sort_index()
    return pivot_df


def prepare_return(df):
    '''
    prepare_return:
    - Parameters:
        + df: DataFrame
    - Output: Return annualized expected returns, covariance matrix, tradings
            days and log returns
    '''

    # Infer average # of trading days per year
    date_range_days = (df.index[-1] - df.index[0]).days
    total_obs = len(df)

    # Estimate trading days from the data
    trading_days_per_year = round(total_obs / (date_range_days / 365))

    # Compute log return, mean, and variance
    log_ret = np.log(df / df.shift(1)).dropna()

    mu = log_ret.mean() * trading_days_per_year                 # Annualized expected returns
    Sigma = log_ret.cov() * trading_days_per_year               # Annualized covariance matrix

    return mu, Sigma, trading_days_per_year, log_ret

def build_constraints(w, mu, allow_short = False, min_return = None):
    '''
    build_constraints
    - Parameters:
        + w: Decision variables
        + mu: expected returns
        + allow_short: T/F if we allow short selling
        + min_return: minimum number of expected returns
    - Output: return a list of contraints for the optimization problem
    '''

    constraints = [cp.sum(w) == 1]

    # If not allow short selling
    if not allow_short:
        constraints.append(w >= 0)

    # If allow minimum return
    if min_return is not None:
        constraints.append(mu.values @ w >= min_return)
    return constraints

def calculate_portfolio_stats(w_val, mu, Sigma, rf, log_ret, trading_days):
    '''
    calculate_portfolio_stats:
    - Parameters:
        + w_val: value for decision variables
        + mu: mean returns
        + Sigma: covariance matrix
        + rf: risk-free rate
        + log_ret: log return
        + trading_days: number of trading days
    - Output: return the stats for evaluating optimization model
    '''

    # Expected return
    exp_return = float(mu.values @ w_val)

    # Volatility
    vol = float(np.sqrt(w_val @ Sigma.values @ w_val))

    # Sharpe ratio
    sharpe = (exp_return - rf) / vol

    # Daily portfolio returns
    port_daily_ret = log_ret @ w_val
    VaR_5 = np.percentile(port_daily_ret, 5) * trading_days
    CVaR_5 = port_daily_ret[port_daily_ret <= np.percentile(port_daily_ret, 5)].mean() * trading_days

    return exp_return, vol, sharpe, VaR_5, CVaR_5

def optimize_min_variance(df, capital, rf = 0.04, allow_short = False, min_return = None, verbose = False):
    '''
    optimize_min_variance:
    - Parameters:
        + df: DataFrame
        + capital: amount of money for investment
        + rf: risk-free rate
        + allow_short: allow short selling
        + min_return: minimum return requirement
        + verbose: T/F if we want to print output result
    - Output: Return the result along with the stats of the model
    '''

    # Computer log returns and covariance matrix
    mu, Sigma, trading_days, log_ret = prepare_return(df)

    # Solve Minimum-Variance Portfolio
    tickers = df.columns.tolist()
    n = len(tickers)
    w = cp.Variable(n)

    constraints = build_constraints(w, mu, allow_short, min_return)

    # Objective: minimize risk
    objective = cp.Minimize(cp.quad_form(w, Sigma.values))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed. Please check model again!")
    
    w_val = w.value.round(4)
    # Get the latest price
    latest_prices = df.iloc[-1]

    # Allocate the money & calculate share
    alloc_usd = capital * w_val
    alloc_shares = (alloc_usd / latest_prices).round(2)

    # Populate to DataFrame
    result = pd.DataFrame({
        'Weight': w_val,
        'Dollars': alloc_usd.round(2),
        'Shares': alloc_shares
    }, index=tickers)

    exp_ret, vol, sharpe, VaR, CVaR = calculate_portfolio_stats(w_val, mu, Sigma, rf, log_ret, trading_days)

    if verbose:
        print(result)
        print(f"\nExpected return: {exp_ret:.2%}")
        print(f"Volatility     : {vol:.2%}")
        print(f"Sharpe ratio   : {sharpe:.2f}")
        print(f"VaR (5%)       : -{abs(VaR):.2%}")
        print(f"CVaR (5%)      : -{abs(CVaR):.2%}")
    
    return {
        "allocations": result,
        "expected_return": exp_ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "VaR_5": VaR,
        "CVaR_5": CVaR
    } 

def optimize_max_return(df, capital, rf = 0.04, allow_short = False, max_volatility = None, verbose = False):
    '''
    optimize_max_return:
    - Parameters:
        + df: DataFrame
        + capital: amount of money for investment
        + rf: risk-free rate
        + allow_short: allow short selling
        + max_volatility: maximum risk tolerant
        + verbose: T/F if we want to print output result
    - Output: Return the result along with the stats of the model
    '''

    # Computer log returns and covariance matrix
    mu, Sigma, trading_days, log_ret = prepare_return(df)

    # Solve Minimum-Variance Portfolio
    tickers = df.columns.tolist()
    n = len(tickers)
    w = cp.Variable(n)

    constraints = build_constraints(w, mu, allow_short)
    if max_volatility is not None:
        constraints.append(cp.quad_form(w, Sigma.values) <= max_volatility ** 2)

    # Objective: maximize return
    objective = cp.Maximize(mu.values @ w)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed. Please check model again!")
    
    w_val = w.value.round(4)
    # Get the latest price
    latest_prices = df.iloc[-1]

    # Allocate the money & calculate share
    alloc_usd = capital * w_val
    alloc_shares = (alloc_usd / latest_prices).round(2)

    # Populate to DataFrame
    result = pd.DataFrame({
        'Weight': w_val,
        'Dollars': alloc_usd.round(2),
        'Shares': alloc_shares
    }, index=tickers)

    exp_ret, vol, sharpe, VaR, CVaR = calculate_portfolio_stats(w_val, mu, Sigma, rf, log_ret, trading_days)

    if verbose:
        print(result)
        print(f"\nExpected return: {exp_ret:.2%}")
        print(f"Volatility     : {vol:.2%}")
        print(f"Sharpe ratio   : {sharpe:.2f}")
        print(f"VaR (5%)       : -{abs(VaR):.2%}")
        print(f"CVaR (5%)      : -{abs(CVaR):.2%}")
    
    return {
        "allocations": result,
        "expected_return": exp_ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "VaR_5": VaR,
        "CVaR_5": CVaR
    } 

if __name__ == "__main__":
    df = pd.read_csv('../data/stock_data_2023-01-01_to_2025-06-29.csv')
    # Pivot the table
    # Get some example tickers for testing
    tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'WMT', 'MSFT', 'GOOGL', 'TSLA']

    df = prep_data(df, tickers)
    
    result_min = optimize_min_variance(df, capital = 20000, verbose=True)
    result_max = optimize_max_return(df, capital = 20000, max_volatility=0.2, verbose=True)
    # print("Hello Word!")