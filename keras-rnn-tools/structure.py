import pandas as pd
import sklearn

class keras_tools:
	def __init__(self, data:pd.DataFrame, \
										n_y_vals:int = 1,
										scaler = None,
										data_orientation = 'row',
										debug:bool = False):
		"""Setup the keras-rnn-tools helper class with passed variables
				
				Args:
					data (pd.DataFrame): base dataframe in pandas format.
					n_y_vals (int): The number of y values to capture for each data set.
					scaler (sklearn scaler or string, optional): optional scaling of data, passed as sklearn scaler or string of scaler type
					debug (bool): indication of whether to output print values for debugging
		"""


		self.data = data
		self.n_y_vals = n_y_vals
		self.scaler = scaler
		self.debug = debug


		if data_orientation == 'row':
			if self.debug == True: print("Row-wise orientation")
		elif data_orientation == 'column':
			if self.debug == True: print("Column-wise orientation")
		else:
			raise AttributeError(f"Type {data_orientation} specified is not valid. Must be either 'column' or 'row'")
		
		if isinstance(self.scaler, str):
			if self.debug == True: print("scaler string")
			if 'minmax' in self.scaler.lower():
				from sklearn.preprocessing import MinMaxScaler
				self.scaler = MinMaxScaler()

			elif 'standard' in self.scaler.lower():
				from sklearn.preprocessing import StandardScaler
				self.scaler = StandardScaler()
			else:
				raise AttributeError("Invalid Scaler Type Passed (minmax or standard expected)")
		elif self.scaler is not None:
			if self.debug == True: print("scaler passed")
		else:
			if self.debug == True: print("no scaler passed")


	def train_test_split(self, 
										split_type:str = 'sample',
										split_pct:float = 0.7,
										val_split_pct:float = 0.1,
										output_scaler:bool = False,
										fill_na:bool = True,
										return_as_df:bool = False):
		"""Create the base train-test-validation split for time-series data
				
				Args:
					split_type (str): Indication of the type of split to perform. Must be one of 'sequential', 'overlap', or 'sample'
					split_pct (bool, optional): 
					val_split_pct (bool, optional): 
					output_scaler (bool, optional): Include the fit scaler in the output. Default is False.
					fill_na (bool): Replace all NAs with 0's, typical prep. Default is True.
					return_as_df (bool): Option to instead return the data as a dataframe (useful for debugging). Default is False.
				Returns:

		"""
		# basic parameter checking
		if split_pct < 0 or split_pct > 1:
			raise AttributeError(f"split_pct must be between 0 and 1. {split_pct} passed.") 
		if val_split_pct < 0 or val_split_pct > 1:
			raise AttributeError(f"val_split_pct must be between 0 and 1. {val_split_pct} passed.") 

		if fill_na==True:
			self.data.fillna(0, inplace=True)

		if split_type == 'sequential':
			if self.debug == True: print("sequential split")
		elif split_type == 'overlap':
			if self.debug == True: print("overlap split")
		elif split_type == 'sample':
			if self.debug == True: print("sample split")
		else:
			raise AttributeError(f"Type {split_type} specified is not valid")
		if self.debug == True: print(self.data)