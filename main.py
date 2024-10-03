#!/usr/bin/env python
# coding=utf-8
"""
@contributor: PanLianggang
@contact: 1304412077@qq.com
@file: main.py
@date: 2024/10/3 14:47
Class Description:
- Briefly describe the purpose of this class here.
@license: MIT
"""
from common.util.dataset_dealer import Dataset
from common.util.data_reader.EmailEuCoreDataset import EmailEuCoreDataset

# 读取数据集和truthtable（如有）
a = Dataset(EmailEuCoreDataset)
raw_data, truthtable = a.read()
print(1)
# 调用算法


# 返回结果，包括运行时间，正确率，可视化网络等

