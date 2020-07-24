import pandas as pd
import numpy as np
# import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# libraries used for train test split
import random
from math import floor
from types import SimpleNamespace

class keras_tools:
	def __init__(self, data:pd.DataFrame, \
										y_val = None,
										ts_n_y_vals:int = None,
										data_orientation:str = 'row',
										debug:bool = False):
		"""Setup the keras-rnn-tools helper class with passed variables
				
				Args:
					data (pd.DataFrame): base dataframe in pandas format.
					y_val (str or pd.DataFrame, optional)
					ts_n_y_vals (int): The number of y values to capture for each data set.
					data_orientation (string): string specifying whether the data frame that is passed will need to be pivoted or not ('row' or 'column', row gets transposed for time-series problems)
					debug (bool): indication of whether to output print values for debugging
		"""


		self.data = data
		self.debug = debug

		# check if y_val is populated
		if y_val is not None:
			
			self.y_val = y_val #TODO: add logic to split on y value
			
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
		
		# split df data
		self.train_df = ""
		self.test_df = ""
		self.valid_df = ""
		
		# transformed data as np
		self.X_train = ""
		self.X_test = ""
		self.X_valid = ""
		self.y_train = ""
		self.y_test = ""
		self.y_valid = ""
		
		

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
		self.scaler.transform(self.X_valid)
		self.scaler.transform(self.y_train)
		self.scaler.transform(self.y_test)
		self.scaler.transform(self.y_valid)
			
		if output_scaler: return self.scaler
		
		
	def _chunk_data (self, df, scaler = None, output_labels = True, **kwargs):
      

	    # reassign the dictionary variables to variables in namespace for easy access
	    n = SimpleNamespace(**kwargs)
	    np_arr_list, y_arr_list = [], []
	    
	            
	    # loop through each step and create a new np array to add to list
	    for chunk_start in range(n.start, (n.end - n.sample_size + 1), n.step):
	        
	        # get a chunk of x values and store to array
	        print("From {} to {}".format(chunk_start, chunk_start + n.sample_size))
	        np_chunk = np.array(df.iloc[:,(chunk_start):(chunk_start + n.sample_size)])
	        # add stored array to list
	        np_arr_list.append(np_chunk)
	        
	        if output_labels:
	            print("Y samples from {} to {}".format((chunk_start + n.sample_size), (chunk_start + n.sample_size + n.y_size)))
	            y_df_chunk = df.iloc[:,(chunk_start + n.sample_size):(chunk_start + n.sample_size + n.y_size)]
	            y_np_chunk = np.array(y_df_chunk)
	            y_arr_list.append(y_np_chunk)
	        
	    # stack all the x samples together
	    np_stacked_chunks = np.stack(np_arr_list)
	    x_reshaped = np.transpose(np_stacked_chunks, (0,2,1))
	    
	    if output_labels:
	        # stack all the y samples together
	        y_np_stacked_chunks = np.stack(y_arr_list)
	        y_reshaped = y_np_stacked_chunks
	        return x_reshaped, y_reshaped
	    else:
	        return x_reshaped
	        
	def _seq_split(self):
		pass

	def train_test_split(self, 
										split_type:str = 'sample',
										split_pct:float = 0.3,
										val_split_pct:float = 0.1,
										fill_na:bool = True,
										return_df:bool = False):
		"""Create the base train-test-validation split for time-series data
				
				Args:
					split_type (str): Indication of the type of split to perform. Must be one of 'sequential', 'overlap', or 'sample'
					split_pct (bool): 
					val_split_pct (bool, optional): 
					fill_na (bool): Replace all NAs with 0's, typical prep. Default is True.
					return_df (bool): Option to instead return the data as a dataframe (useful for debugging). Default is False.
				Returns:

		"""
		#### basic parameter checking
		if split_pct < 0 or split_pct > 1:
			raise AttributeError(f"split_pct must be between 0 and 1. {split_pct} passed.") 
		if val_split_pct < 0 or val_split_pct > 1:
			raise AttributeError(f"val_split_pct must be between 0 and 1. {val_split_pct} passed.") 

		if fill_na==True:
			self.data.fillna(0, inplace=True)

		#### create split depending on split_type
		if split_type == 'sequential':
			if self.debug == True: print("sequential split")
			
			train_test_split_num = floor(self.data.shape[1] * (1 - split_pct - val_split_pct))
			test_val_split = floor(self.data.shape[1] * (1 - val_split_pct))
			if self.debug == True: print("Split at {} and {}".format(train_test_split_num, test_val_split))

			# print(self.data, train_test_split_num)
			
			self.train_df = self.data.iloc[:, 0:train_test_split_num]
			self.test_df = self.data.iloc[:, train_test_split_num:test_val_split]
			
			if val_split_pct > 0 and val_split_pct < 1:
				# create validation variables
				x_val_start = test_val_split
				x_val_end = self.data.shape[1] - self.ts_n_y_vals
				
				self.valid_df = self.data.iloc[:, test_val_split:]
				if return_df: return self.train_df, self.test_df, self.valid_df
			else:
				if return_df: return self.train_df, self.test_df
			 
			
		elif split_type == 'overlap':
			if self.debug == True: print("overlap split")
		elif split_type == 'sample':
			if self.debug == True: print("sample split")
			
			# try to split by y_val first, move on if it's not set
			try:
				X_train, X_test, y_train, y_test = train_test_split(self.data, self.y_val, test_size=split_pct)
				X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split_pct/(1-split_pct))
			except AttributeError:
				# for time-series this method only works if you want to sample specific features and keep the full time-series
				
				# split out test_df then remove rows from train_df
				self.test_df = self.data.loc[self.data.sample(frac=split_pct, replace=False).index]
				self.train_df = self.data.loc[~self.data.index.isin(self.test_df.index)]
				
				# split out valid_df then remove the rows from train_df
				self.valid_df = self.train_df.loc[self.train_df.sample(frac=val_split_pct, replace=False).index]
				self.train_df = self.train_df.loc[~self.train_df.index.isin(self.valid_df.index)]
				
				# split_num, train_split_list, test_split_list, val_split_list = [], [], [], []
				# n_y_vals = self.data.shape[1]
				# #TODO: add seed
				# for x in range(n_y_vals):
				# 	rand_num = random.uniform(0,1)
				# 	split_num.append(rand_num)
				# 	test_split_list.append(rand_num < split_pct)
				# 	val_split_list.append((rand_num - split_pct) < val_split_pct and rand_num >= split_pct)
					
				# print(split_num)
				# print(val_split_list)
					
				# train_split_list = [not c for c in test_split_list]
				# self.X_train = np.array(self.data.iloc[:,train_split_list])
				# self.X_test = np.array(self.data.iloc[:,test_split_list])
				# self.X_valid = np.array(self.data.iloc[:,val_split_list])
				
				# print(f"train: {self.X_train.shape}; test: {self.X_test.shape}; valid: {self.X_valid.shape}")
				
			# self.X_train = np.array(self.data)[1:]
			return self.X_train
		else:
			raise AttributeError(f"Type {split_type} specified is not valid")
		if self.debug == True: print(self.data)

	def transform_for_rnn(self, 
							step:int = 1,
							sample_size:int = 1):
		"""Transforms split data into format needed for RNN
				
				Args:
					split_type (str): Indication of the type of split to perform. Must be one of 'sequential', 'overlap', or 'sample'
					step (int): The number of steps before you take another sample (e.g. [1,3,5,6,7,9] and step of 2 would return x values of [[1,3][5,6][7,9]])
					sample_size (int): The number of samples you want to take for each value (e.g. [1,3,5,6,7,9] and sample_size of 3 would return x values of [[1,3,5][3,5,6][5,6,7][6,7,9]])
					return_as_df (bool): Option to instead return the data as a dataframe (useful for debugging). Default is False.
				Returns:

		"""
		# x_end = train_test_split_num - self.ts_n_y_vals
		# #         print("x_end: {}".format(x_end))
		# # create test variables
		# x_test_start = train_test_split_num
		# x_test_end = test_val_split - self.ts_n_y_vals
		# # run the process on the training data
		# x_reshaped, y_reshaped = self._chunk_data(self.data, start = 0, end = x_end, step = step, sample_size = sample_size, y_size = self.ts_n_y_vals)
		
		# if self.debug == True: print("split here for test range {} to {}".format(x_test_start, x_test_end))
		# # get test data
		# x_reshaped_test, y_reshaped_test = self._chunk_data(self.data, start = x_test_start, end = x_test_end, step = step, sample_size = sample_size, y_size = self.ts_n_y_vals)
		
		
		# self.X_train = x_reshaped
		# self.y_train = y_reshaped
		# self.X_test = x_reshaped_test
		# self.y_test = y_reshaped_test
		
		# if val_split_pct > 0 and val_split_pct < 1:
		# 	if self.debug == True: print("split here for val range {} to {}".format(x_val_start, x_val_end))
		# 	# create val data sets
		# 	x_reshaped_val, y_reshaped_val = self._chunk_data(self.data, start = x_val_start, end = x_val_end, step = step, sample_size = sample_size, y_size = self.ts_n_y_vals)
			
		# 	self.X_valid = x_reshaped_val
		# 	self.y_valid = y_reshaped_val
		pass

	def get_input_shape(self, parameter_list):
		pass

	def unscale(self, parameter_list):
		pass

	def model_summary(self):
		pass