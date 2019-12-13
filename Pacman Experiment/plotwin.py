# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file is the visualization of win rate

# Author: Quan Chen


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

######## Small Grid Case: #########

data1 = pd.read_csv('./pacq_win.txt', header = None)
data2 = pd.read_csv('./appro_win.txt', header = None)
data3 = pd.read_csv('./dual_win.txt',header = None)
data4 = pd.read_csv('./appro_simple_win.txt',header = None)
plt.figure(1)
plt.plot(data1,label = 'Normal Q Learning')
plt.plot(data2,label = 'Approximate Q Identity Extractor')
plt.plot(data3,label = 'Double Q')
plt.plot(data4,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Number of win cases out of 10')
plt.title('Comparison among Approaches')
plt.legend()
plt.show()


####### Medium Grid Case ###########
data5 = pd.read_csv('./big_q_win.txt', header = None)
data6 = pd.read_csv('./big_appro_win.txt', header = None)
data7 = pd.read_csv('./big_dual_win.txt',header = None)
data8 = pd.read_csv('./big_appro_simple_win.txt',header = None)

plt.figure(2)
plt.plot(data5,label = 'Normal Q Learning')
plt.plot(data6,label = 'Approximate Q Identity Extractor')
plt.plot(data7,label = 'Double Q')
plt.plot(data8,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Number of win cases out of 10')
plt.title('Comparison among Approaches')
plt.legend()
plt.show()
