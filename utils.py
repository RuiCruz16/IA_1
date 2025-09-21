import numpy as np
from loader import stocksRecords
RISK_FREE_RATE = -0.05

# Calculates the daily returns of a given stock based on its adjusted closing prices.
# The return is calculated as the percentage change between consecutive days.
# If the previous adjusted close is 0, the return is set to 0 to avoid division by zero.
def calculate_stock_return(stock):
    res = []
    for i in range(1, len(stocksRecords[stock])):
        current = stocksRecords[stock][i]
        previous = stocksRecords[stock][i - 1]
        if previous.adj_close == 0:
            daily_return = 0
        else:
            daily_return = (current.adj_close / previous.adj_close) - 1
        res.append(daily_return)

    return np.array(res)


# Calculates the mean (average) return of a given stock.
# It filters out non-finite values (e.g., NaN or infinity) before calculating the mean.
# If there are no valid returns, it returns 0.
def calculate_stock_mean_return(stock):
    returns = calculate_stock_return(stock)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0
    else:
        return np.sum(returns) / len(returns)


# Represents a stock with its name and mean return.
# The mean return is calculated using the `calculate_stock_mean_return` function.
class Stock:
  def __init__(self, name):
    self.name = name
    self.stock_return = calculate_stock_mean_return(name)
  def __repr__(self):
    return f"{self.name} || {self.stock_return}"
  def __eq__(self, __o: object) -> bool:
      return self.id == __o.id and self.name == __o.name


stocks = {}
for stockName in stocksRecords.keys():
    stock = Stock(stockName)
    stocks[stockName] = stock


# Calculates the mean (average) return of a portfolio.
# The portfolio is a dictionary where keys are stock names and values are their weights.
# The mean return is calculated as the weighted sum of the mean returns of the stocks in the portfolio.
def calculate_portfolio_mean_return(portfolio):
    total_return = 0
    for stock in portfolio:
        stockObj = stocks[stock]
        stock_return = stockObj.stock_return
        if np.isfinite(stock_return):
            total_return += stock_return * portfolio[stock]

    return total_return


# Calculates the Sharpe ratio of a portfolio.
# The Sharpe ratio is a measure of risk-adjusted return, calculated as:
# (Portfolio Mean Return - Risk-Free Rate) / Portfolio Standard Deviation.
# If the standard deviation is too small, it returns negative infinity to indicate high risk.
def calculate_portfolio_sharpe_ratio(portfolio):
    average_return = calculate_portfolio_mean_return(portfolio)
    standard_deviation = calculate_portfolio_standard_deviation(portfolio)
    if standard_deviation < 1e-6:
        return -np.inf
    else:
        return (average_return - RISK_FREE_RATE) / standard_deviation


# Calculates the standard deviation of a portfolio's returns.
# The standard deviation is a measure of the portfolio's risk (volatility).
# It aligns the returns of all stocks in the portfolio to the same time period and calculates the portfolio's overall volatility.
def calculate_portfolio_standard_deviation(portfolio):
    weights = np.array([portfolio[stock] for stock in portfolio])
    returns_list = [calculate_stock_return(stock) for stock in portfolio]

    if not returns_list or all(np.all(np.isnan(r)) for r in returns_list):
        return 0

    min_length = min(len(r) for r in returns_list)
    aligned_returns = np.column_stack([r[:min_length] for r in returns_list])

    if np.all(np.isnan(aligned_returns)):
        return 0

    aligned_returns = aligned_returns[~np.isnan(aligned_returns).any(axis=1)]

    if aligned_returns.size == 0:
        return 0

    portfolio_returns = np.dot(aligned_returns, weights)
    return np.std(portfolio_returns, ddof=1)
