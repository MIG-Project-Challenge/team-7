import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()

def wwma(values, n):
    return values.ewm(alpha=1/n, adjust=False).mean()

def ATR(high, low, close, timeperiod):
    data = pd.DataFrame()
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, timeperiod)
    return atr

def BBANDS(data, timeperiod, nbdevup, nbdevdn, matype=None):
    sma = data.rolling(timeperiod).mean()
    std = data.rolling(timeperiod).std()
    bollinger_up = sma + std * nbdevup
    bollinger_down = sma - std * nbdevdn
    return bollinger_up, sma, bollinger_down

def SMA(data, timeperiod):
    sma = data.rolling(window=timeperiod).mean()
    return sma

def EMA(data, timeperiod):
    ema = data.ewm(span=timeperiod,adjust=False).mean()
    return ema

def STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype):
    high = high.rolling(fastk_period).max()
    low = low.rolling(fastk_period).min()

    fastk = ((close - low) / (high - low)) * 100
    fastd = fastk.rolling(slowk_period).mean()
    slowk = fastd.rolling(slowk_period).mean()

    if slowd_matype == 0:
        slowd = slowk.rolling(slowd_period).mean()
    else:
        slowd = slowk.rolling(slowd_period).apply(lambda x: np.convolve(x, np.ones(slowd_period), mode='valid') / slowd_period)

    return slowk, slowd

def RSI(data, timeperiod):
    delta = data.diff()
    delta = delta[1:]

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def CCI(high, low, close, timeperiod):
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(timeperiod).mean()
    mean_deviation = np.abs(typical_price - sma).rolling(timeperiod).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def MACD(data, fastperiod, slowperiod, signalperiod):
    exp1 = data.ewm(span=fastperiod, adjust=False).mean()
    exp2 = data.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram