import os

files = os.listdir('data/grasp-anywhere/seen/positive_grasp')

idxs = dict()

for file in files:
    idx = file.split('_')[0]
    if idx not in idxs:
        idxs[idx] = None

# print(idxs)
for file in idxs:
    with open('need.txt', 'a+') as f:
        f.writelines(file+'\n')