
import os
root_dir = '/home/iasl-orin2/Desktop/experimental/' # your path here where to parent directory where repos are
local_dir = '/home/iasl-orin2/Desktop/local/'
os.chdir(root_dir)
import sys
sys.path.append(root_dir)
import benchmarks.benchmark_methods as bm


def dummy(nums):
	total = 0
	for num in nums:
		total *= num
	return total

powers, times, temperatures, returns = bm.run_benchmark(10, dummy, [i for i in range(100)])

print(powers)
print(times)
print(temperatures)
print(returns)run
