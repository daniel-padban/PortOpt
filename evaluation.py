from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from yfinance import Tickers

class Evaluation(object):
    def __init__(self, algo:object,tickers:Tickers, start_eval:datetime,end_eval:datetime,lookback:timedelta, eval_period:timedelta,):
        '''
        Class to evaluate portfolio optimization algorithms on spans of time with rebalancing periods

        :param algo: algorithm to evaluate - callable with signature algo(returns:pd.DataFrame, (start:datetime.datetime, end:datetime.datetime), **kwargs) -> weights
        '''
        
        self.algo = algo
        self.tickers = tickers

        self.start_eval = start_eval
        self.end_eval = end_eval
        self.lookback = lookback    
        self.eval_period = eval_period

    def _periodize(self) -> list[tuple[tuple,tuple]]:
        '''
        Create a list of tuples with the start and end dates for each evaluation period and the lookback period

        :return: list of tuples with the start and end dates for each evaluation period and the lookback period, shape:( (lookback_start, lookback_end), (eval_start, eval_end) )
        '''
        periods:list[(tuple,tuple)] = []

        start = self.start_eval
        end = self.end_eval

        eval_start:datetime = start+self.lookback
        eval_end=timedelta(days=0)
        while eval_start+ self.eval_period <= end:
            lookback_start = eval_start - self.lookback
            lookback_end = eval_start
            eval_end = eval_start + self.eval_period
            periods.append(((lookback_start,lookback_end),(eval_start, eval_end)))
            
            eval_start = eval_end
            

        return periods
    
    def _prices2returns(self, prices:pd.DataFrame):
        '''
        Convert prices to log returns
        
        :param prices: DataFrame of prices
        :return log_returns: DataFrame of log returns
        '''

        log_returns = pd.DataFrame(np.log(prices / prices.shift(1)))
        log_returns = log_returns.dropna()
        log_returns = log_returns.rename(columns={'Close': 'log_returns'})

        return log_returns
    
    def _returns(self):
        '''
        Get data for the evaluation and lookback periods
        '''
        prices = self.tickers.history(start=self.start_eval, end=self.end_eval)['Close']
        returns = self._prices2returns(prices)
        
        return returns
    
    def _optimize_weights(self, returns, **algo_kwargs):
        '''
        Optimizes asset weights
                        
        :param returns: DataFrame of returns
        :param algo_kwargs: keyword arguments for the algorithm
        :param lookback_period: tuple with the start and end dates for the lookback period
        :return weights: optimized weights

        '''
        
        weights = self.algo(returns, **algo_kwargs)
        # Assuming algo has a method to optimize weights
        # This method should return the optimized weights

        return weights

    def compare_returns(self, **algo_kwargs):
        '''
        Evaluates the algo on the test data.

        :return optimized_returns: optimized returns 
        :return comp_returns: comparison returns
        '''
        all_returns = self._returns()
        evals = []
        periods = self._periodize()
        for period in periods:
            period_returns = all_returns.loc[period[0][0]:period[0][1]]
            weights = self._optimize_weights(period_returns,**algo_kwargs)
            eval_returns = all_returns.loc[period[1][0]:period[1][1]] @ weights
            evals.append(eval_returns)
        optimized_returns = pd.concat(evals)

        comp_returns = all_returns.loc[periods[0][1][0]:periods[-1][1][1]].mean(axis=1)

        return optimized_returns, comp_returns

    
if __name__ == '__main__':
    from allocator import WeightOptimizer
    # Example usage
    start = datetime(2020, 1, 1)
    end = datetime(2023, 1, 1)
    lookback = timedelta(days=2*365)
    eval_period = timedelta(days=30*6)

    tickers = Tickers('AAPL MSFT')
    allocator = WeightOptimizer(10000,1e-3,2,0.025,timedelta(365),0.1)
    evaluator = Evaluation(allocator.optimize_weights, tickers, start, end, lookback, eval_period)

    opt_returns, comp_returns = evaluator.compare_returns(alpha=0.3, beta=0.1, gamma=0.6)
    opt_returns.head()