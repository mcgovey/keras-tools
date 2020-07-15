from .context import KerasTools

import pandas as pd


class TestRNN:
    def setup(self):
        self.sales_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv')
        
        self.helper = ""
    
    def test_util(self):
        
        
        self.helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = 28, debug=False)
        self.helper.train_test_split(split_type='sequential')
        
        
        assert self.helper.ts_n_y_vals == 28



### Tests
## train_test_split
# split_pct less than 0
# split_pct greater than 1
# val_split_pct less than 0
# val_split_pct greater than 1

## initialization
# ts_n_y_vals
# y_val as string
# y_val as df