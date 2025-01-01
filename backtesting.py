import numpy as np
import backtrader as bt
from trading_strategy import AdvancedTradingStrategy

def backtest(model, env, X, y):
    obs = env.reset()
    done = False
    portfolio_values = [10000]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(env.portfolio_value)
    
    return {
        'portfolio_values': portfolio_values,
        'returns': np.diff(portfolio_values) / portfolio_values[:-1]
    }

def run_backtest(data, initial_cash=100000.0, commission=0.001):
    cerebro = bt.Cerebro()
    
    # Add data feed to Cerebro
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    # Add strategy to Cerebro
    cerebro.addstrategy(AdvancedTradingStrategy)
    
    # Set our desired cash start
    cerebro.broker.setcash(initial_cash)
    
    # Set the commission
    cerebro.broker.setcommission(commission=commission)
    
    # Run the backtest
    results = cerebro.run()
    
    # Get final portfolio value
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    
    return results, final_value, pnl

