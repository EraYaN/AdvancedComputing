import os

d='../run_output'

paths = [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

for path in paths:
    image = ''
    with open(os.path.join(path,'run_settings.ini'),'r') as run_settings:
        #TODO parse image filename
    with open(os.path.join(path,'stderr.log'),'r') as stderr:
        #TODO parse nvprof output
    with open(os.path.join(path,'stdout.log'),'r') as stdout:
        #TODO parse benchmark output
