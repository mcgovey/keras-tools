# from unittest import TestCase

import structure as ktt

import numpy as np
import pandas as pd

sales_df = pd.read_csv('../sales_train_validation.csv')

helper = ktt.keras_tools(sales_df, n_y_vals = 28, debug=True)

helper.train_test_split()

# class TestJoke(TestCase):
#     def test_is_string(self):
#         s = funniest.joke()
#         self.assertTrue(isinstance(s, basestring))
