#!/usr/bin/env python
# coding=utf-8
import os
import pickle

pkl_data_dir = r"D:\code\FYP\CommunityDetection\data\temp"


def save_pkl_to_temp(data, filename):
    if not os.path.exists(pkl_data_dir):
        os.makedirs(pkl_data_dir)
    filename = os.path.join(pkl_data_dir, filename + ".pkl")

    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"数据已成功保存为{filename}l")


def read_pkl_from_temp(filename):
    filename = os.path.join(pkl_data_dir, filename + ".pkl")
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data
