# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file is the visualization of rewards results

# Author: Quan Chen


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

######## Small Grid Case: #########

data1 = pd.read_csv('./pacq.txt', header = None)
data2 = pd.read_csv('./appro.txt', header = None)
data3 = pd.read_csv('./dual.txt',header = None)
data4 = pd.read_csv('./appro_simple.txt',header = None)
plt.figure(1)
plt.plot(data1,label = 'Normal Q Learning')
plt.plot(data2,label = 'Approximate Q Identity Extractor')
plt.plot(data3,label = 'Double Q')
plt.plot(data4,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Comparison among Approaches')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(data1,label = 'Normal Q Learning')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Normal Q Learning Approach')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(data2,label = 'Approximate Q Learning')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Approximate Q Learning Approach')
plt.legend()
plt.show()

plt.figure(4)
plt.plot(data3,label = 'Double Q')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Double Q')
plt.legend()
plt.show()

plt.figure(5)
plt.plot(data4,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Approximate Q Simple Extractor')
plt.legend()
plt.show()


####### Medium Grid Case ###########

data5 = pd.read_csv('./big_q.txt', header = None)
data6 = pd.read_csv('./big_appro.txt', header = None)
data7 = pd.read_csv('./big_dual.txt',header = None)
data8 = pd.read_csv('./big_appro_simple.txt',header = None)


plt.figure(6)
plt.plot(data5,label = 'Normal Q Learning')
plt.plot(data6,label = 'Approximate Q Identity Extractor')
plt.plot(data7,label = 'Double Q')
plt.plot(data8,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Comparison among Approaches')
plt.legend()
plt.show()


plt.figure(7)
plt.plot(data5,label = 'Normal Q Learning')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Normal Q Learning Approach')
plt.legend()
plt.show()

plt.figure(8)
plt.plot(data6,label = 'Approximate Q Learning')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Approximate Q Learning Approach')
plt.legend()
plt.show()

plt.figure(9)
plt.plot(data7,label = 'Double Q')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Double Q')
plt.legend()
plt.show()

plt.figure(10)
plt.plot(data8,label = 'Approximate Q Simple Extractor')
plt.xlabel('Number of Episodes / 100')
plt.ylabel('Average Rewards per 100 Episodes')
plt.title('Approximate Q Simple Extractor')
plt.legend()
plt.show()