try:
   import cPickle as pickle
except:
   import pickle
import matplotlib

from WrapperShared import Variant
from terminaltables import AsciiTable
from operator import itemgetter

results = []
display_results = []

# load results from Wrapper
print("Loading SSEvsAVX.pickle file.")
with open('SSEvsAVX.pickle', 'rb') as handle:
    results = pickle.load(handle)

# this test code should print the exact output as Wrapper.py
table_heading = ['platform',
    'type',
    'config',
    'threads',
    "data_size",
    "iterations",
    "variant",
    "sequential_time",
    "variant_time",
    "relative_improvement",
    "had_error"]
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
    9:"right",
    10:"left"
}
#Sort by improvement
#results.sort(key=attrgetter('relative_improvement'), reverse=True)

#Sort by time
results.sort(key=itemgetter('variant_time'))

display_results.append(table_heading)
for result in results:
    error_text = "Yes" if result["had_error"] else "No"
    display_results.append([result['platform'],
    result['type']['name'],
    result['config'],
    result['threads'],
    result["data_size"],
    result["iterations"],
    result["variant"].name,
    "{0:.5f} us".format(result["sequential_time"] * 1000000 / result["iterations"]),
    "{0:.5f} us".format(result["variant_time"] * 1000000 / result["iterations"]),
    "{0:.3%}".format(result["relative_improvement"]),
    error_text])



results_table = AsciiTable(display_results)
results_table.justify_columns = table_justify
print(results_table.table)