# physical-ai-hw3
NYCU Physical AI 2025 Fall

In your original physical-ai directory, `git pull` to get new `hw3` directory.

# hw3 Robot Manipulation

## Installation

Follow the installation instruction in https://github.com/google-research/ravens

1. Create and activate Conda environment, then install GCC and Python packages.

```shell
git pull
cd hw3
cd ravens
conda create --name pdm-hw3 python=3.7 -y
conda activate pdm-hw3
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
```
2. Install GPU acceleration with NVIDIA CUDA 10.1 and cuDNN 7.6.5 for Tensorflow.
```bash
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```
## Task 1

Implement your_fk function in fk.py
- execution example
```bash
python fk.py
pybullet build time: Sep 22 2020 00:55:20
============================ Task 1 : Forward Kinematic ============================

- Testcase file : fk_test_case_ta1.json
- Your Score Of Forward Kinematic : 5.000 / 5.000, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 5.000 / 5.000, Error Count :    0 /  100

- Testcase file : fk_test_case_ta2.json
- Your Score Of Forward Kinematic : 5.000 / 5.000, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 5.000 / 5.000, Error Count :    0 /  100

====================================================================================
- Your Total Score : 20.000 / 20.000
====================================================================================
```

## Task 2

Implement your_ik function in ik.py
- execution example
```bash
python ik.py
###################################### log #########################################

============================ Task 2 : Inverse Kinematic ============================

- Testcase file : ik_test_case_ta1.json
- Mean Error : 0.001048
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

- Testcase file : ik_test_case_ta2.json
- Mean Error : 0.001482
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

====================================================================================
- Your Total Score : 40.000 / 40.000
====================================================================================
```

## Task 3

Test your ik implementation in the Transporter Networks 's frame work by inferencing "Block Insertion Task" on a mini set of 10 testing data

**Step 1.** Download dataset at https://drive.google.com/file/d/1Jh8hAvraT1Zt1YfSNRT_lMJXbsK4Wcse/view?usp=sharing and put the whole `block-insertion-easy-test/` folder under `hw3/ravens/`

**Step 2.** Download checkpoint at https://drive.google.com/file/d/1cmFbqTzuu6IUJPlx1eOq2djRSubfM94H/view?usp=sharing and put the whole `checkpoints/` folder under `hw3/ravens/`

**Step 3.** Testing the model and your ik implementation 
- execution example
 ```shell
cd ravens
CUDA_VISIBLE_DEVICES=-1 python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion-easy --agent=transporter --n_demos=1000 --n_steps=20000# No need to use GPU
###################################### log ###########################################

Loading pre-trained model at 20000 iterations.
============================ Task 3 : Transporter Network ============================

Test: 1/10
Instructions for updating:
Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
Total Reward: 1.0 Done: True
Test: 2/10
Total Reward: 1.0 Done: True
Test: 3/10
Total Reward: 1.0 Done: True
Test: 4/10
Total Reward: 1.0 Done: True
Test: 5/10
Total Reward: 1.0 Done: True
Test: 6/10
Total Reward: 1.0 Done: True
Test: 7/10
Total Reward: 1.0 Done: True
Test: 8/10
Total Reward: 1.0 Done: True
Test: 9/10
Total Reward: 1.0 Done: True
Test: 10/10
Total Reward: 1.0 Done: True
====================================================================================
- Your Total Score : 10.000 / 10.000
====================================================================================
 ```
# Reference
https://github.com/google-research/ravens
