#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:58:45 2020

@author: bamm
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ll_path = '../mknap/mknap-output-data/ll/mknap_aco_gpu2080_problem'
musket_data = pd.read_csv('../bpp_output_data/musket/result_BPP70_musket.csv', delimiter=',', header=None)

# extract data from files
Musket1024Average = [0, 0, 0, 0, 0, 0]
Musket2048Average = [0, 0, 0, 0, 0, 0]
Musket4096Average = [0, 0, 0, 0, 0, 0]
Musket8192Average = [0, 0, 0, 0, 0, 0]

MusketAverage = [Musket1024Average, Musket2048Average, Musket4096Average, Musket8192Average]

ll1024Average = [0, 0, 0, 0, 0, 0]
ll2048Average = [0, 0, 0, 0, 0, 0]
ll4096Average = [0, 0, 0, 0, 0, 0]
ll8192Average = [0, 0, 0, 0, 0, 0]

lowlevelAverage = [ll1024Average, ll2048Average, ll4096Average, ll8192Average]

# iterate over files
for f in range(20):

    file_path_str = ll_path + str(f) + ".out"

    breno_data = pd.read_csv(file_path_str, delimiter=',', header=None)

    for x in range(4):
        ma = 0
        lla = 0

        for y in range(4):
            lla = lla + breno_data.at[((x * 4) + y), 4]

            m_index = ((f * 40) + ((x * 10) + y))

        ma = musket_data[f][x + 1]
        lla = lla / 10

        MusketAverage[x][f] = ma
        lowlevelAverage[x][f] = lla
