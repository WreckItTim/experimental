import utils.global_methods as gm # sys.path.append('path/to/parent/repo/')
from jtop import jtop
import numpy as np
from time import time

def run_benchmark(n_runs, func, func_params):
	# repeat results to capture variance
	times = [None] * n_runs
	jetson_stats = [None] * n_runs
	returns = [None] * n_runs
	for run in range(n_runs):  
	    sw = Stopwatch()
	    jetson = jtop() # create benchmark logging thread
	    jetson.start() # start benchmark window
	    ret = func(**func_params)
	    jetson.close() # close benchmark logging thread
	    times[run] = sw.lap()
	    returns[run] = ret
	    jetson_stats[run] = jetson.stats.copy()
	return jetson_stats, times, returns
