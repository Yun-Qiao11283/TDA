import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

tickers = ["AAPL", "NVDA", "TSLA", "MSFT"]
data = yf.download(tickers, period="1y")
print(data['Close'])

data['Close'].plot()
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
