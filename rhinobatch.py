# -*- coding: utf-8 -*-
"""
@author: Krzysztof Zaborski
V1.0 August 2019
PyMarine
Module starting Rhino
"""
#%% IMPORT
import subprocess
import tkinter as tk

def start_batch(): 
       subprocess.call([r'C:\Program Files\Rhino 6\System\Rhino.exe'])
    return 0
