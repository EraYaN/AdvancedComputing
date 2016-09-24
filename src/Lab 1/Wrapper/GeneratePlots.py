try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.pyplot as plt
import numpy as np

from WrapperShared import Variant
from terminaltables import AsciiTable
from operator import itemgetter

results = []

### Part A
## Task 1
# most suitable amount of threads
# PAR SEQ comparison
    # fixed data size (2048)
    # sweep threads
    # plot speedup vs threads at fixed data size

# load results from Wrapper
#OpenMP = {
#    'name':'OpenMP',
#    'variants':[Variant.base],
#    'configs':['Release'],
#    'data_sizes':[2048],
#    'thread_range':np.arange(1, 65, 1)
#}
print("Loading partA_task1.pickle file.")
with open('partA_task1.pickle', 'rb') as handle:
    results = pickle.load(handle)

graphX = []
graphY = []

for result in results:
    graphX.append(result['threads'])
    graphY.append(result['relative_improvement'])

f = plt.figure(1)
plt.bar(graphX,graphY,align='center')
plt.ylim([np.min(graphY)-(np.max(graphY)-np.min(graphY))*0.01,np.max(graphY)+(np.max(graphY)-np.min(graphY))*0.01])

plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.title('OpenMP vs normal execution (n=2048)')
plt.grid(True)
plt.savefig("test.png")
f.show()


## Task 2
# sweep data size 10 - 10000
# plot speedup vs data size at Task1 threads

# load results from Wrapper
print("Loading partA_task2.pickle file.")
with open('partA_task2.pickle', 'rb') as handle:
    results = pickle.load(handle)

graphX = []
graphY = []

for result in results:
    graphX.append(result['data_size'])
    graphY.append(result['relative_improvement'])

g = plt.figure(2)
plt.bar(graphX,graphY,align='center')
plt.ylim([np.min(graphY)-(np.max(graphY)-np.min(graphY))*0.01,np.max(graphY)+(np.max(graphY)-np.min(graphY))*0.01])

plt.xlabel('Data Size')
plt.ylabel('Speedup')
plt.title('OpenMP vs normal execution (threads=40)')
plt.grid(True)
plt.savefig("test1.png")
g.show()


### Part B
## Task 1
# sweep matrix size
# plot speedup vs size

# load results from Wrapper
print("Loading partB_task1.pickle file.")
with open('partB_task1.pickle', 'rb') as handle:
    results = pickle.load(handle)

graphX = []
graphY = []

for result in results:
    graphX.append(result['data_size'])
    graphY.append(result['relative_improvement'])

h = plt.figure(3)
plt.bar(graphX,graphY,align='center')
plt.ylim([np.min(graphY)-(np.max(graphY)-np.min(graphY))*0.01,np.max(graphY)+(np.max(graphY)-np.min(graphY))*0.01])

plt.xlabel('Matrix Vector Size')
plt.ylabel('Speedup')
plt.title('OpenMP vs normal execution (threads=40)')
plt.grid(True)
plt.savefig("test.png")
h.show()