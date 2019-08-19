# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
General functions module
"""
#%% IMPORT
import math
import os
import numpy as np

#%% Remove first row from a .csv file
def csv_remove_first_row(source, destination, delete=False):
    with open(source,'r') as f:
        with open(destination,'w') as f1:
            next(f) # skip header line
            for line in f:
                f1.write(line)
    if delete:
        os.remove(source)
    return 0
