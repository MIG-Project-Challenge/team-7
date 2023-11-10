# Import necessary libraries
from eval_algo import eval_actions
from collections import defaultdict
import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import math

# Function to calculate Average True Range (ATR)
def calculateATR(data, timeperiod):
    # Calculating True Range components
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Computing ATR using rolling mean
    atr = tr.rolling(timeperiod).mean()
    return atr

# Function to calculate Supertrend
def calculateSupertrend(data, atr_period=10, multiplier=3):
    data_copy = data.copy()  # Made a copy to avoid SettingWithCopyWarning
    data_copy['ATR'] = calculateATR(data_copy, atr_period)

    # Calculation of basic upper and lower bands for Supertrend
    data_copy['basic_ub'] = (data_copy['High'] + data_copy['Low']) / 2 + multiplier * data_copy['ATR']
    data_copy['basic_lb'] = (data_copy['High'] + data_copy['Low']) / 2 - multiplier * data_copy['ATR']
    data_copy['final_ub'] = data_copy['basic_ub']
    data_copy['final_lb'] = data_copy['basic_lb']

    # Loop through data to compute Supertrend trend values
    trend = []
    for i in range(len(data_copy)):
        if i == 0:  # Handling the initial data point
            trend.append(0)
        elif data_copy.at[data_copy.index[i], 'Close'] <= data_copy.at[data_copy.index[i], 'final_ub']:
            trend.append(1)
        elif data_copy.at[data_copy.index[i], 'Close'] >= data_copy.at[data_copy.index[i], 'final_lb']:
            trend.append(-1)
        else:
            if trend[-1] == 1:
                trend.append(1)
            else:
                trend.append(-1)
    return trend

# Class for Supertrend algorithm
class Algo:
    def __init__(self, data_path, cash=25000, atr_period=10, multiplier=3):
        # Read and preprocess data
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index(["Ticker", "Date"], inplace=True)

        # Extract open prices and initialize various parameters
        self.open_prices = self.df['Open'].unstack().values
        self.trades = np.zeros(self.open_prices.shape)
        self.cash = cash
        self.positions = [0] * len(self.open_prices)
        self.short_positions = defaultdict(list)
        self.port_values = [0] * len(self.open_prices[0])
        self.port_values[0] = self.cash
        self.atr_period = atr_period
        self.multiplier = multiplier

    # Method to run Supertrend strategy
    def runSupertrend(self):
        for stock in range(len(self.open_prices)):
            data = self.df.xs(self.df.index.get_level_values('Ticker').unique()[stock]).copy()  # Made a copy
            trend = calculateSupertrend(data, self.atr_period, self.multiplier)

            # Loop through data to execute trades based on Supertrend signals
            for day in range(1, len(self.open_prices[0]) - 1):
                self.port_values[day] = self.calcPortfolioValue(day)

                if trend[day] == 1 and trend[day - 1] == -1:
                    self.trades[stock][day + 1] = 1
                    self.handleBuy(stock, day + 1)
                elif trend[day] == -1 and trend[day - 1] == 1:
                    self.trades[stock][day + 1] = -1
                    self.handleSell(stock, day + 1)
                else:
                    self.trades[stock][day + 1] = 0

        self.port_values[-1] = self.calcPortfolioValue(len(self.open_prices[0]) - 1)

    def saveTrades(self, path):
        # for convention please name the file "trades.npy"
        np.save(path, self.trades)

    def handleBuy(self, stock, day):
        # Calculate the number of shares to buy based on available cash
        price = self.open_prices[stock][day]
        numShares = math.floor(int(self.cash / price))

        if numShares > 0:
            self.cash -= price * numShares
            self.positions[stock] += numShares

    def handleSell(self, stock, day):
        # Calculate the number of shares to sell based on current position
        price = self.open_prices[stock][day]
        numShares = abs(self.positions[stock])

        if numShares > 0:
            self.cash += price * numShares
            self.positions[stock] -= numShares

    def cashValid(self):
        return self.cash >= 0
    
    def calcShortValue(self, day):
        # calculates value of all the short positions on 'day'
        # note that this value can be positive or negative
        val = 0
        for stock in self.short_positions.keys():
            for short_price, short_amount in self.short_positions[stock]:
                val += (short_price - self.open_prices[stock][day]) * short_amount

        return val
    
    def calcPortfolioValue(self, day):
        # the sum of cash and positive positions (and their value on 'day') + short positions value
        # cash + long positions + short positions

        value = self.cash
        for stock in range(len(self.open_prices)):
            if self.positions[stock] > 0:
                value += self.open_prices[stock][day] * self.positions[stock]
        return value + self.calcShortValue(day)

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supertrend Algorithm")
    parser.add_argument("-p", "--prices", help="Path to stock prices csv file")
    prices_path = parser.parse_args().prices

    if prices_path is None:
        print("Please provide a path to a stock prices csv file using: main_algo.py -p <path to file>")
        exit(1)

    algo = Algo(prices_path)

    algo.runSupertrend()

    # Evaluate trades
    port_values, sharpe_ratio = eval_actions(algo.trades, algo.open_prices, cash=25000, verbose=True)

    print(sharpe_ratio)
    print(algo.port_values[-1])

    algo.saveTrades("trades.npy")