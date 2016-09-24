import pickle
import matplotlib

from enum import Enum
from terminaltables import AsciiTable

# load results from Wrapper
with open('results.pickle', 'rb') as handle:
    results = pickle.load(handle)

### Part A
## Task 1
# most suitable amount of threads
    # fixed data size (512?)
    # sweep threads
    # find shortest OpenMP time
    # plot time vs threads at fixed data size
# PAR SEQ comparison
    # plot speedup vs threads at fixed data size

## Task 2
# sweep data size 10 - 10000
# plot speedup vs data size at Task1 threads

# this test code should print the exact output as Wrapper.py

table_heading = [
    'platform',
    'type',
    'config',
    'threads',
    "data_size",
    "iterations",
    "variant",
    "sequential_time",
    "variant_time",
    "relative_improvement"
]
table_justify = {
    0:'left',
    1:'left',
    2:'left',
    3:'left',
    4:"right",
    5:"right",
    6:"left",
    7:"right",
    8:"right",
    9:"right"
}

display_results.append(table_heading)
for result in results:
    display_results.append([
    result['platform'],
    result['type']['name'],
    result['config'],
    result['threads'],
    result["data_size"],
    result["iterations"],
    result["variant"].name,
    "{0:.3e}".format(result["sequential_time"]),
    "{0:.3e}".format(result["variant_time"]),
    "{0:.3%}".format(result["relative_improvement"])
])

results_table = AsciiTable(display_results)
results_table.justify_columns=table_justify;
print(results_table.table)