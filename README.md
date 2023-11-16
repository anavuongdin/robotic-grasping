# Antipodal Robotic Grasping

[Updated] Weights are available at this [link](https://drive.google.com/file/d/15WDMUKaQ2tUJgiXAqij47NwcxQZ4c801/view?usp=sharing)

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/anavuongdin/robotic-grasping.git
```

- Create a virtual environment
```bash
$ conda create -n grasping python=3.9
```

- Activate the virtual environment
```bash
$ conda activate grasping
```

- Install the requirements
```bash
$ cd robotic-grasping
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Weights
- All weights are stored at `weights/model_<dataset>`.

## Inference example
- An atom example is shown in `run_robotic_exp.py`. To run this file:

```bash
$ python run_robotic_exp.py --weight weights/model_<dataset>
```

- Note, to run with diffusion model, please run the following command:

```bash
$ python run_robotic_exp.py --weight /path/to/weight --diffusion True
```
Specifically, diffusion model takes some extra parameters to proceed, it takes six variables, however, two of them are None, two are constants, and the rest is the same as other baselines. Please refer to L75-77, for further information. In general, the input for all baselines are the same.

## Output structure

L67 of the file `run_robotic_exp.py` prints the output structure. For simplicity, assume the image size is 224 x 224 (note, we can use any resolution we like). There are four components of output:
- pos_pred: An array of [1, 224, 224], each number in the array (0/1) indicates that pixel is in the predicted grasp pose or not. Note that, it should be rounded to the nearest number (0, 1) as the prediction is usually in a continuous domain.
- cos_pred/sin_pred: An array of [1, 224, 224], each number in the array indicates the angular of that pixel corresponding to the grasp pose.
- width_pred: An array of [1, 224, 224], each number in the array indicates the width corresponding to the grasp pose.

For a clearer view of the output structure, please check the file `utils/dataset_processing/grasp.py` (L252-259). These lines show how to convert from grasp poses to output structure. I hope you can base on this information to revert the output structure to the grasp poses.

Please contact me if you have any questions. Thank you for your time. Best regards, An.
