# -*- coding: utf-8 -*-
"""
A small script to turn a pd.read_csv line to a from ___ import ___ line 

Is there a better way to do this? Probably, but this works
"""

import pandas as pd

multi_phase = pd.read_csv("example_data/multi.csv")
single_phase = pd.read_csv("example_data/single.csv")