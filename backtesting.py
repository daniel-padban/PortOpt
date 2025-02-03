from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from allocator import WeightOptimizer


class BackTester():
    def __init__(self,test_weights,start:date,end:date,tickers:yf.Tickers,comp_weights=None,):
        '''
        
        :param comp_weights: weight of assets to compare against
        '''

        self.asset_data:pd.DataFrame = tickers.history(start=start,end=end,auto_adjust=True)['Close']
        daily_returns = self.asset_data.pct_change(1,).dropna()
        daily_returns.dropna(inplace=True)

        if comp_weights is not None:
            self.comparison_portfolio_returns = daily_returns.dot(comp_weights)
        else:
            self.comparison_portfolio_returns = None
        self.test_portfolio_returns = daily_returns.dot(test_weights)
        print('TESTING')


    def period_return(self,returns:pd.DataFrame,intervals_per_period=252):
        total_comp_return = (1+returns).prod()
        num_intervals = returns.shape[0]
        period_return = (total_comp_return ** (intervals_per_period/num_intervals))-1
        return period_return
    
    def cumulative_returns(self,returns:pd.DataFrame):
        cumulative_returns = np.exp(returns.cumsum()) - 1
        return cumulative_returns
    
    def period_sharpe(self,returns:pd.DataFrame,risk_free_annual:float,intervals_per_period=252):
        avg_return = returns.mean(0)
        std = returns.std(0)

        daily_risk_free = (1+risk_free_annual)**(1/intervals_per_period)-1
        excess_return = avg_return-daily_risk_free

        sharpe = (excess_return*intervals_per_period-daily_risk_free)/std
        return sharpe


class repeatingBacktester():
    def __init__(self, 
                current_comp:set,
                changes_path:str,
                eval_start:date, 
                eval_end:date,
                lookback:int|timedelta,
                rebalancing:int|timedelta,
                alpha:float,
                beta:float,
                gamma:float,
                rf:float,
                rf_period:timedelta,
                num_iter:int,
                weight_decay:float,
                lr,):
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_iter = num_iter
        self.lr = lr
        self.rf = rf
        self.rf_period = rf_period
        self.weight_decay = weight_decay

        pass

    def _register_changes(self):
        changes_df = pd.read_csv(self.changes_path)
        

    def _optimize(self,num_assets,returns):
        allocator = WeightOptimizer(self.num_iter,self.lr,num_assets,self.rf,self.rf_period,self.weight_decay)
        allocator.optimize_weights(self.alpha,self.beta,self.gamma,returns)
