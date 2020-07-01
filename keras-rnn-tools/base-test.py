# from unittest import TestCase

import structure as krt

import numpy as np
import pandas as pd

sales_df = pd.read_csv('../sales_train_validation.csv')

helper = krt.rnn_helper(sales_df, n_y_vals = 28)

helper.train_test_split()

# class TestJoke(TestCase):
#     def test_is_string(self):
#         s = funniest.joke()
#         self.assertTrue(isinstance(s, basestring))
