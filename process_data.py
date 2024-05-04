import pickle
import os
import numpy as np
from pathlib import Path

grasp_instructions_2_lv = ["Pick up {} by its {}.",
                      "Hold {} at its {}.",
                      "Grab {} on its {}.",
                      "Grasp {} at its {}.",
                      "Take hold of {} on its {}.",
                      "Lift {} by its {}.",
                      "Take {} by its {}.",
                      "Fetch {} by its {}.",
                      "Grip {} on its {}."]

grasp_instructions_1_lv = ["Grasp me the {}.",
                           "Give me the {}",
                           "Hand me the {}",
                           "Pass me the {}.",
                           "Bring me the {}.",]

with open("dataset.pkl", 'rb') as f:
    part_vocab = pickle.load(f)

image_path = "data/grasp-anything++/unseen/image"
file_path = "data/grasp-anything++/unseen/prompt"
grasp_label = "data/grasp-anything++/unseen/grasp_label"
grasp_instruction_path = "data/grasp-anything++/unseen/grasp_instructions"

grasp_files = os.listdir(grasp_label)

files = os.listdir(file_path)
image_files = os.listdir(image_path)
counter = 0

for file in grasp_files:
    img_file = file.split('_')[0] + ".jpg"
    if img_file not in image_files:
        os.remove(os.path.join(grasp_label, file))

for file in files:
    with open(os.path.join(file_path, file), 'rb') as f:
        id = file.split(".pkl")[0]
        image_id = id + ".jpg"
        # print(image_id)
        if image_id not in image_files:
            continue
        
        counter += 1
        # print(id)
        prompt, queries = pickle.load(f)
        for i, query in enumerate(queries):
            for j, part in enumerate(query):
                fn = "{}_{}_{}.pt".format(id, i, j)
                fn_path = Path(os.path.join(grasp_label, fn))

                # Check if the file exists
                if fn_path.exists():
                    # print("The file exists.")
                    grasp_instruction_fn = "{}_{}_{}.pkl".format(id, i, j)
                    with open(os.path.join(grasp_instruction_path, grasp_instruction_fn), 'wb') as f:
                        rand = np.random.random()
                        if rand < 0.8:
                            instr = np.random.choice(grasp_instructions_2_lv).format(query, part_vocab[query][j])
                        else:
                            instr = np.random.choice(grasp_instructions_1_lv).format(query)
                        # print(instr)
                        pickle.dump(instr, f)
                else:
                    # print("The file does not exist.")
                    pass

        #     print(part_vocab[query])

# print(counter)