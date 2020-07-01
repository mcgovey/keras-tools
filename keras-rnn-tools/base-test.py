# from unittest import TestCase

import structure as krt

import numpy as np
import pandas as pd

sales_df = pd.read_csv('../sales_train_validation.csv')

krt.train_test_split(data = sales_df)

# class TestJoke(TestCase):
#     def test_is_string(self):
#         s = funniest.joke()
#         self.assertTrue(isinstance(s, basestring))
