import os

results_dir = 'results'

results = os.listdir(results_dir)
results = sorted(results)
for rdir in results:
    with open(os.path.join(results_dir, rdir, rdir + '.txt'), 'r') as f:
        for i,l in enumerate(f):
            if i == 0:
                meta = l[:-1]
            if i == 1:
                notes = l[:-1]
    print(rdir)
    print(' ' * 4, meta)
    print(' ' * 4, notes, '\n')