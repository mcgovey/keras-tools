# from unittest import TestCase

import structure as ktt

# import numpy as np
import pandas as pd


import unittest

class TestStringMethods(unittest.TestCase):
	def setUp(self):
		self.sales_df = pd.read_csv('../sales_train_validation.csv')


	def test_equality(self):
		self.helper = ktt.keras_tools(self.sales_df, n_y_vals = 28, debug=False)

		self.helper.train_test_split()
		
		self.assertEqual(self.helper.n_y_vals, 28)



if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
	unittest.TextTestRunner(verbosity=2).run(suite)