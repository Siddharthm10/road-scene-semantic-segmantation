import torch
import numpy as np

# Raw to Train ID mapping based on Cityscapes documentation
id_to_trainid = {
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 2,    # building
    10: 3,   # wall
    11: 4,   # fence
    12: 5,   # pole
    13: 6,   # traffic light
    14: 7,   # traffic sign
    15: 8,   # vegetation
    16: 9,   # terrain
    17: 10,  # sky
    18: 11,  # person
    19: 12,  # rider
    20: 13,  # car
    21: 14,  # truck
    22: 15,  # bus
    23: 16,  # train
    24: 17,  # motorcycle
    25: 18,  # bicycle
    26: 255, # license plate (ignore)
    27: 255, # out of range
    28: 255, # out of range
    29: 255, # out of range
    30: 255, # out of range
    31: 255, # out of range
    32: 255, # out of range
    33: 255  # out of range
}

# Function to convert raw labels to train labels
def convert_id_to_train_id(label):
    label = label.clone()  # Clone to avoid modifying the original tensor
    for k, v in id_to_trainid.items():
        label[label == k] = v
    return label

