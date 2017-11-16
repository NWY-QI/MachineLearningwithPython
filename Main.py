from feature_functions import *
import numpy as np
import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go

# (1) Load up our data & create moving average (previous video)
df = pd.read_csv('data/EURUSDHour.csv')
df.columns = [['date','open','high','low','close','AskVol']]
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df= df[['open','high','low','close','AskVol']]
df['Symbol'] = 'EURUSD'
df = df.drop_duplicates(keep=False)
df = df.iloc[:500]

ma = df.close.rolling(center=False,window=30).mean ()


#(2) Get function data from selected function:

# # Video - 3 Heiken Ashi
# HAresults = heikenashi(df,[1])
# HA = HAresults.candles[1]

# # Video - 4 Detrending
# detrended = detrend(df, method='difference')

# # Video - 5 Fourier & Sine series
# f = sine(df,[10,15],method='difference')

# # Video - 6 WADL
# results = wadl(df,[15])
# line = results.wadl[15]

# # Video - 7 Finish up
# * Resampled
resampled = OHLCresample(df, '15H')
print(resampled)
resampled.index = resampled.index.droplevel(0)

# * Momentum function
# m = momentum(df,[10])
# res = m.close[10]

# * stochastic function
# s = stochastic(df,[14,15])
# res = s.close[14]

# * Williams function
# w = williams(df,[15])
# res = w.close[15]

# * PROC function
# p = proc(df,[30])
# res = p.proc[30]

# * adosc function
# AD = adosc(df,[30])
# res = AD.AD[30]

# * macd function
# m = macd(df,[15,30])
# res = m.signal

# * CCI function
# c = cci(df,[30])
# res = c.cci[30]

# * Bollinger bands function
# b = bollinger(df,[20],2)
# res = b.bands[20]

# * Price average function
# avs = paverage(df,[20])
# res = avs.avs[20]

# * Slopes function
# s = slopes(df,[20])
# res = s.slope[20]


#(3) Plot
trace0 = go.Ohlc(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name='Currency Quote')
trace1 = go.Scatter(x=df.index,y=ma)
# trace2 = go.Ohlc(x=HA.index,open=HA.open,high=HA.high,low=HA.low,close=HA.close,name='Heiken Ashi')
# trace2 = go.Scatter(x=df.index,y=detrended)
# trace2 = go.Scatter(x=line.index,y=line.close)
trace2 = go.Ohlc(x=resampled.index.to_pydatetime(),open=resampled.open,high=resampled.high,low=resampled.low,
                close=resampled.close,name='Currency Quote')
# trace2 = go.Scatter(x=res.index, y=res.close)
# trace2 = go.Scatter(x=res.index, y=res.K)
# trace2 = go.Scatter(x=res.index, y=res.R)
# trace2 = go.Scatter(x=res.index, y=res.close)
# trace2 = go.Scatter(x=res.index, y=res.AD)
# trace2 = go.Scatter(x=res.index, y=res.SL)
# trace2 = go.Scatter(x=res.index, y=res.close)
# trace2 = go.Scatter(x=res.index, y=res.upper)
# trace2 = go.Scatter(x=res.index, y=res.close)
# trace2 = go.Scatter(x=res.index, y=res.high)


data = [trace0,trace1,trace2]

fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)

py.offline.plot(fig,filename='Main.html')

# print(df.head())

