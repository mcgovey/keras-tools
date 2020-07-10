# from unittest import TestCase

import structure as ktt

# import numpy as np
import pandas as pd


import unittest

class TestStringMethods(unittest.TestCase):
	def setUp(self):
		self.sales_df = pd.read_csv('../sales_train_validation.csv')


	def test_equality(self):
		self.helper = ktt.keras_tools(self.sales_df, ts_n_y_vals = 28, debug=False)

		self.helper.train_test_split(split_type='sequential')
		
		self.assertEqual(self.helper.ts_n_y_vals, 28)

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

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
	unittest.TextTestRunner(verbosity=2).run(suite)