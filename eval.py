import sys
from math import log

# occurance of prefix in dataset
lines1 = open(sys.argv[1]).readlines()
lines2 = open(sys.argv[2]).readlines()

# drop partial lines
lines2 = lines2[: (len(lines2) / 16) * 16]
lines1 = lines1[:len(lines2) / 16]
prefix = map(float, lines1)

# occurance of predicted completeion in dataset
method = map(float, lines2)

# group 16 numbers together, because we predict top-16 queries (some of them might be 0)
group = lambda x: [x[i*16:(i+1)*16] for i in range(len(x)/16)]

# sum the occurance of each group
method = map(sum, group(method))

# some prefix will appear 0 times in dataset (for example, the query is too long and truncated)
# in that case both x and y are 0. Avoid dividing by 0.
prob = [x*1./(y + 1e-5) for (x,y) in zip(method, prefix)]
prob = filter( lambda x: x<=1, prob)

print max(prob)
mean = lambda x: sum(x)*1./len(x)
print 'mean prob =', mean(prob), 'mean hit =', mean(method)/16
