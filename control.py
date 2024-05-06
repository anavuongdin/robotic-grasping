import os
import pickle

# files = os.listdir('data/grasp-anywhere/unseen/positive_grasp')

# idxs = dict()

# for file in files:
#     idx = file.split('_')[0]
#     if idx not in idxs:
#         idxs[idx] = None

# # print(idxs)
# for file in idxs:
#     with open('need.txt', 'a+') as f:
#         f.writelines(file+'\n')

idxs = []
with open("good_seen.txt", 'r') as f:
    idxs_str = f.readlines()
    for idx in idxs_str:
        idxs.append(idx.strip('\n'))

with open("rest_seen.txt", 'r') as f:
    idxs_str = f.readlines()
    for idx in idxs_str:
        idxs.append(idx.strip('\n'))

with open("split/grasp-anything/seen.obj", 'rb') as f:
    data = pickle.load(f)
    # print(data)
    new_data = []
    for file in data:
        idx = file.split('_')[0]
        
        if idx not in idxs:
            new_data.append(file)
    
with open("split/grasp-anything++/train/seen.obj", 'wb') as f:
    pickle.dump(new_data, f)