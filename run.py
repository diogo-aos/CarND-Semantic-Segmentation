import subprocess as sp
import sys

learning = [0.001]
epochs = [50]
bsize = [10]
dropout = [0.5]
notes = ''

print('estimated model sizes: ', 2.1 * len(learning) * len(epochs), ' GB')
y = input('continue? (y/n)')
if y != 'y':
    sys.exit(1)

for l in learning:
    for e in epochs:
        for d in dropout:
            p = sp.Popen(['/home/diogoaos/p2/bin/python', 'main2.py', '-E', e, '-L', l, '-B', b, '-D', d, '-M', ,'-N', notes])
            p.wait()