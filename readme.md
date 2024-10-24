# Community Detection
这里是一个社区发现相关的代码整理仓库，用于记录我的毕设项目

## 环境
python 3.11.9
Microsoft Windows 10 家庭版

## 数据集介绍

所有数据集在读取的时候会做额外处理
1. 所有节点编号都从0开始
2. 所有社区编号都从0开始

| 名字                       | url                                                    | 基础信息                     |
| ------------------------ | ------------------------------------------------------ | ------------------------ |
| email-Eu-core network    | https://snap.stanford.edu/data/email-Eu-core.html      | 非重叠社区 1005点 25571边 42社区  |
| Zachary's Karate Club    | https://en.wikipedia.org/wiki/Zachary%27s_karate_club | 非重叠社区 34个节点 78条边 2社区     |
| cora                     | https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz | 非重叠社区 2708个节点 5429条边 8社区 |
| Social circles: Facebook | https://snap.stanford.edu/data/ego-Facebook.html       | 4039点 88234边             |

## 算法介绍
？