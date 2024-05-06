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
with open("good_unseen.txt", 'r') as f:
    idxs_str = f.readlines()
    for idx in idxs_str:
        idxs.append(idx.strip('\n'))

with open("rest_unseen.txt", 'r') as f:
    idxs_str = f.readlines()
    for idx in idxs_str:
        idxs.append(idx.strip('\n'))

with open("split/grasp-anything/unseen.obj", 'rb') as f:
    data = pickle.load(f)
    # print(data)
    new_data = []
    for file in data:
        idx = file.split('_')[0]
        
        if idx in idxs:
            new_data.append(file)
    
with open("split/grasp-anything++/test/unseen.obj", 'wb') as f:
    pickle.dump(new_data, f)