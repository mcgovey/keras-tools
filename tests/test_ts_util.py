from .context import KerasTools

import pandas as pd
import pytest

class TestRNN:
    def setup(self):
        self.sales_df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv').iloc[:100,:]
        
        self.helper = ""
        self.y_steps = 5
    
    def test_split(self):
        
        
        self.helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = self.y_steps, debug=False)
        self.helper.train_test_split(split_type='sequential')
        
        
        assert self.helper.ts_n_y_vals == self.y_steps

    def test_scale(self):
        self.scale_helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = self.y_steps, debug=False)
        
        
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
        self.scale_helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = self.y_steps, debug=False)
        
        split_pct = 0.3
        val_split_pct = 0.1
    
        self.scale_helper.train_test_split(split_type='sequential',
										split_pct = split_pct,
										val_split_pct = val_split_pct)
        
        assert self.scale_helper.train_df.shape == (self.sales_df.shape[1], (1 - split_pct - val_split_pct) * self.sales_df.shape[0])
        assert self.scale_helper.test_df.shape == (self.sales_df.shape[1], split_pct * self.sales_df.shape[0])
        assert self.scale_helper.valid_df.shape == (self.sales_df.shape[1], val_split_pct * self.sales_df.shape[0])
        
    # def test_seq_sample(self):
    #     self.scale_helper = KerasTools.keras_tools(self.sales_df, ts_n_y_vals = self.y_steps, debug=False)
        
        
    #     step = 1
    #     sample_size = 1
        
        
    #     self.scale_helper.rnn_transform(
				# 						step = step,
				# 						sample_size = sample_size)
										
        
    #     assert self.scale_helper.X_train.shape == ((1 - split_pct - val_split_pct) * self.sales_df.shape[0] - self.y_steps, sample_size, self.sales_df.shape[1])
    #     assert self.scale_helper.y_train.shape == ((1 - split_pct - val_split_pct) * self.sales_df.shape[0] - self.y_steps, self.sales_df.shape[1], self.y_steps)
    #     assert self.scale_helper.X_test.shape == (split_pct * self.sales_df.shape[0] - self.y_steps, sample_size, self.sales_df.shape[1])
    #     assert self.scale_helper.y_test.shape == (split_pct * self.sales_df.shape[0] - self.y_steps, self.sales_df.shape[1], self.y_steps)
    #     assert self.scale_helper.X_valid.shape == (val_split_pct * self.sales_df.shape[0] - self.y_steps, sample_size, self.sales_df.shape[1])
    #     assert self.scale_helper.y_valid.shape == (val_split_pct * self.sales_df.shape[0] - self.y_steps, self.sales_df.shape[1], self.y_steps)
        

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