# Community Detection
这里是一个社区发现相关的代码整理仓库，用于记录我的毕设项目

## 环境
python 3.11.9
Microsoft Windows 10 家庭版

## 数据集介绍

所有数据集在读取的时候会做额外处理
1. 所有节点编号都从0开始
2. 所有社区编号都从0开始

| 名字                                                                | 是否重叠  | 节点数量    | 边数量     | 社区数量  | 笔记链接                      | url                                                                                                                                                                             |
| ----------------------------------------------------------------- | ----- | ------- | ------- | ----- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| email-Eu-core network                                             | 非重叠社区 | 1005    | 25571   | 42    | [[斯坦福_邮件数据]]              | https://snap.stanford.edu/data/email-Eu-core.html                                                                                                                               |
| Social circles: Facebook                                          |       | 4039    | 88234   |       | [[斯坦福_脸书数据]]              | https://snap.stanford.edu/data/ego-Facebook.html                                                                                                                                |
| Zachary's Karate Club                                             | 非重叠社区 | 34      | 78      | 2     | [[Zachary's Karate Club]] | https://en.wikipedia.org/wiki/Zachary%27s_karate_club                                                                                                                           |
| cora                                                              | 非重叠社区 | 2708    | 5429    | 8     | [[cora]]                  | https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz                                                                                                                             |
| DBLP collaboration network and ground-truth communities           | 重叠社区  | 3170080 | 1049866 | 13477 | [[DBLP]]                  | https://snap.stanford.edu/data/com-DBLP.html                                                                                                                                    |
| Political Books                                                   | 非重叠社区 | 105     | 441     | 3     | [[political-books]]       | https://networks.skewed.de/net/polbooks<br>https://github.com/melaniewalsh/sample-social-network-datasets/blob/master/sample-datasets/political-books/political-books-nodes.csv |
| Amazon product co-purchasing network and ground-truth communities | 重叠社区  | 334863  | 925872  |       |                           | https://snap.stanford.edu/data/com-Amazon.html                                                                                                                                  |

## 算法介绍
？