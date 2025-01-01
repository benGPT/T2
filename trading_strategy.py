import backtrader as bt

class AdvancedTradingStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('atr_distance', 2),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close, 
                                       period_me1=self.params.macd_fast, 
                                       period_me2=self.params.macd_slow, 
                                       period_signal=self.params.macd_signal)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        self.order = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.params.rsi_oversold and self.macd.macd > self.macd.signal:
                size = int(self.broker.getcash() / self.data.close)
                self.order = self.buy(size=size)
                self.stop_loss = self.data.close - self.params.atr_distance * self.atr
                self.take_profit = self.data.close + self.params.atr_distance * self.atr

        else:
            if self.rsi > self.params.rsi_overbought or self.macd.macd < self.macd.signal:
                self.order = self.close()

        if self.position:
            if self.data.close < self.stop_loss or self.data.close > self.take_profit:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

#the end#

