from .context import KerasTools

import pandas as pd
import pytest

class TestRNN:
    def setup(self):
        self.sales_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv').iloc[:100,:]
        
        self.helper = ""
    
    def test_split(self):
        
        
        self.helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = 5, debug=False)
        self.helper.train_test_split(split_type='sequential')
        
        
        assert self.helper.ts_n_y_vals == 5

    def test_scale(self):
        self.scale_helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = 10, debug=False)
        
        
        with pytest.raises(AttributeError) as excinfo:
            self.scale_helper.scale() #no scaler passed
            
        assert "Scaler type None" in str(excinfo.value)
        
        # created scaler passed minmax
        
        # name of scaler passed
        self.scale_helper.train_test_split(split_type='sample')
        # self.scale_helper.scale(scaler = "minmax")
        
        # self.scale_helper.scale(scaler = "standard")
        
        # return scaler is true
        
    def test_seq_split(self):
        self.scale_helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = 5, debug=False)
        
        
        self.scale_helper.train_test_split(split_type='sequential')
        
        assert self.scale_helper.X_train.shape
        

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