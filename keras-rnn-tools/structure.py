print("loaded")

import pandas as pd
import sklearn

class rnn_helper:
  def __init__(self, data:pd.DataFrame, \
                    n_y_vals:int = 1):
    """Setup the keras-rnn-tools helper class with passed variables
        
        Args:
          data (pd.DataFrame): base dataframe in pandas format.
          n_y_vals (int): The number of y values to capture for each data set.
    """
    self.data = data
    self.n_y_vals = n_y_vals

  def train_test_split(self,
                      overlap:bool = False, \
                      scaler = None) -> pd.DataFrame:
    """Create the base train-test-validation split for time-series data
        
        Args:
          overlap (bool, optional): 
          scaler (sklearn scaler, optional):
    """
    print(self.data)