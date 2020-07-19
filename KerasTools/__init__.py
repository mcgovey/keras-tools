import pandas as pd
import numpy as np
# import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class keras_tools:
	def __init__(self, data:pd.DataFrame, \
										y_val = None,
										ts_n_y_vals:int = None,
										data_orientation = 'row',
										debug:bool = False):
		"""Setup the keras-rnn-tools helper class with passed variables
				
				Args:
					data (pd.DataFrame): base dataframe in pandas format.
					ts_n_y_vals (int): The number of y values to capture for each data set.
					scaler (sklearn scaler or string, optional): optional scaling of data, passed as sklearn scaler or string of scaler type
					debug (bool): indication of whether to output print values for debugging
		"""


		self.data = data
		self.debug = debug

		# check if y_val is populated
		if y_val is not None:
			if isinstance(y_val, str):
				print("passed y string")
			elif isinstance(y_val, pd.DataFrame):
				print("passed data frame")
		# check if ts_n_y_vals is populated
		elif ts_n_y_vals is not None:
			self.ts_n_y_vals = ts_n_y_vals
			if data_orientation == 'row':
				if self.debug == True: print("Row-wise orientation")
				self.data = self.data.T
				if self.debug == True: print(self.data)
				
			elif data_orientation == 'column':
				if self.debug == True: print("Column-wise orientation")
			else:
				raise AttributeError(f"Type {data_orientation} specified is not valid. Must be either 'column' or 'row'")
		# if neither are populated then raise error
		else:
			raise AttributeError("Either y_val or ts_n_y_vals must be populated.")

		# other variables used
		self.scaler = "" # defined in scale()
		
		self.X_train = ""
		self.X_test = ""
		self.X_val = ""
		self.y_train = ""
		self.y_test = ""
		self.y_val = ""
		
		

	def scale(self, 
										scaler = None,
										output_scaler:bool = False):
		"""Scale the data in the data set. Prescribe the same scaling to the test and validation data sets.
				
				Args:
					scaler (sklearn scaler or string, optional): optional scaling of data, passed as sklearn scaler or string of scaler type (minmax or standard).
					output_scaler (bool, optional): Include the fit scaler in the output. Default is False.
		"""
		
		self.scaler = scaler

		if isinstance(self.scaler, str):
			if self.debug == True: print("scaler string")
			if 'minmax' in self.scaler.lower():
				self.scaler = MinMaxScaler()

			elif 'standard' in self.scaler.lower():
				self.scaler = StandardScaler()
			else:
				raise AttributeError("Invalid Scaler Type Passed (minmax or standard expected)")
		elif self.scaler is not None:
			if self.debug == True: print("scaler passed")
		else:
			if self.debug == True: print("no scaler passed")
			raise AttributeError(f"Scaler type {scaler} was not sklearn scaler or string of ('minmax' or 'standard').")
		
		# fit to training data
		self.scaler.fit(self.X_train)
		
		# transform all the data in the data set
		self.scaler.transform(self.X_train)
		self.scaler.transform(self.X_test)
		self.scaler.transform(self.X_val)
		self.scaler.transform(self.y_train)
		self.scaler.transform(self.y_test)
		self.scaler.transform(self.y_val)
			
		if output_scaler: return self.scaler

	def train_test_split(self, 
										split_type:str = 'sample',
										split_pct:float = 0.3,
										val_split_pct:float = 0.1,
										fill_na:bool = True,
										return_as_df:bool = False):
		"""Create the base train-test-validation split for time-series data
				
				Args:
					split_type (str): Indication of the type of split to perform. Must be one of 'sequential', 'overlap', or 'sample'
					split_pct (bool, optional): 
					val_split_pct (bool, optional): 
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
			self.X_train = np.array(self.data)[1:]
			return self.X_train
			# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_pct)
			# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split_pct/(1-split_pct))
			# TODO: add deterministic parameter
		else:
			raise AttributeError(f"Type {split_type} specified is not valid")
		if self.debug == True: print(self.data)

	def transform_for_rnn(self, parameter_list):
		pass

	def get_input_shape(self, parameter_list):
		pass

	def unscale(self, parameter_list):
		pass

	def model_summary(self):
		pass