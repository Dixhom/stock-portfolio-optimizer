
import pandas_datareader.data as web
import scipy.optimize as optimize
from scipy.optimize import NonlinearConstraint
from scipy.stats import linregress
import numpy as np
import pandas as pd

from datetime import datetime, date, timedelta


class DatasetCreation:
    def __init__(self):
        pass

    def __get_start_month(self, first_of_this_month, n_months):
        year = first_of_this_month.year - (n_months // 12)
        month = first_of_this_month.month - (n_months % 12)
        month = (month - 1) % 12 + 1  # start from 1 (Janurary)
        if first_of_this_month.month <= (n_months % 12):
            year -= 1
        start_month = first_of_this_month.replace(year=year, month=month)
        return start_month

    def __get_data_fetch_period(self, n_train_months, n_test_months):
        first_of_this_month = date.today().replace(day=1)
        end_of_last_month = first_of_this_month - timedelta(days=1)
        n_months = n_train_months + n_test_months

        # the beginning of test set
        start_test_month = self.__get_start_month(
            first_of_this_month, n_test_months)

        # the beginning of train set
        start_train_month = self.__get_start_month(
            first_of_this_month, n_months)

        return start_train_month, start_test_month, end_of_last_month

    def __fetch_stock_data(self, ticker_list, start, end):
        # get data (use Close price)
        stock_prices = {tick: web.DataReader(
            tick, data_source='yahoo', start=start, end=end).Close for tick in ticker_list}
        return stock_prices

    def validate_stock_price_period(self, stock_prices, start_train_month):
        error_ticks = [tick for tick, stock in stock_prices.items()
                       if stock.index[0] > start_train_month]
        if error_ticks:
            raise ValueError(f'Insufficient data period in {error_ticks}')

    def create_datasets(self, ticker_list, n_train_months, n_test_months):
        start_train_month, start_test_month, end_of_last_month = self.__get_data_fetch_period(
            n_train_months, n_test_months)

        stock_prices = self.__fetch_stock_data(
            ticker_list, start_train_month, end_of_last_month)

        self.validate_stock_price_period(stock_prices, start_train_month)

        df = pd.DataFrame(stock_prices)
        df_train = df[df.index < start_test_month.strftime('%Y-%m-%d')]
        df_test = df[df.index >= start_test_month.strftime('%Y-%m-%d')]

        return df, df_train, df_test


class MathOpt:
    def __init__(self):
        pass

    def __cost_func(self, proportions, df, alpha):
        # time series profit of the portfolio
        ts = (proportions * df).sum(axis=1)
        # perform linear regression
        x = np.arange(len(ts))
        y = ts.values
        slope, intercept, r, p, se = linregress(x, y)
        # final profit
        profit = slope * (len(ts) - 1)
        # fluctuation of the profit
        line = slope * x + intercept  # regression line
        flat = ts - line  # flatten the slope
        std = flat.diff()[1:].std()
        # total cost
        cost = -profit + alpha * std
        return cost

    def __constr_func(self, x):
        return sum(x)

    def optimize(self, df, alpha):
        nlc = NonlinearConstraint(self.__constr_func, 1, 1)
        bounds = [(0, 1)] * df.shape[1]  # boundary * number of stocks
        result = optimize.differential_evolution(
            self.__cost_func, bounds, args=(df, alpha), constraints=(nlc,))
        return result.x


def optimize_portfolio(ticker_list, alpha):
    n_train_months = 9
    n_test_months = 3

    # create datasets
    dc = DatasetCreation()
    df, df_train, df_test = dc.create_datasets(
        ticker_list, n_train_months, n_test_months)

    # optimize
    mo = MathOpt()
    prop = mo.optimize(df_train, alpha)

    # add portfolio sum
    df['total'] = (df * prop).sum(axis=1)
    test_vline = df_test.index[0]
    return df, prop, test_vline
