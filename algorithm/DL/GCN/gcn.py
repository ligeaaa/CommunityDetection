#!/usr/bin/env python
# coding=utf-8
import math
from datetime import datetime
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
import os

from algorithm.DL.GCN.gcn_model import GCN
from common.util.decorator import time_record


# 1. 数据加载和预处理
def load_data(raw_data, truth_table, device):
    # 获取节点数
    nodes = set([x for edge in raw_data for x in edge])
    num_nodes = max(nodes) + 1

    # 构建边列表
    edge_index = torch.tensor(raw_data, dtype=torch.long).t().contiguous().to(device)

    # 创建标签张量
    y = torch.zeros(num_nodes, dtype=torch.long).to(device)
    for node, community in truth_table:
        y[node] = community

    # 使用度作为初始特征
    degrees = degree(edge_index[0], num_nodes=num_nodes)
    degrees = degrees.view(-1, 1).to(device)  # 将度数转为列向量形状

    # 随机初始化特征 (16维向量)
    random_features = torch.randn(num_nodes, 16, dtype=torch.float).to(device)

    # 使用身份矩阵（one-hot 编码）
    identity_features = torch.eye(num_nodes, dtype=torch.float).to(device)

    # 将度数、随机特征和身份特征拼接起来
    x = torch.cat([degrees, random_features, identity_features], dim=1)

    # 创建图数据对象
    data = Data(x=x, edge_index=edge_index, y=y).to(device)  # 确保 Data 对象在 device 上
    return data


# 2. 训练函数
def train(model, data, optimizer, loss_fn, epochs=100, device='cpu'):
    model.train()
    data = data.to(device)  # 确保数据在正确的 device 上
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')


# 3. 测试函数
def test(model, data, device='cpu'):
    model.eval()
    data = data.to(device)  # 确保数据在正确的 device 上
    with torch.no_grad():
        output = model(data)
        predictions = output.argmax(dim=1)

    # 将预测结果转化为社区划分
    communities = {}
    for node, community in enumerate(predictions.cpu().numpy()):
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    return list(communities.values())


# 4. 模型保存函数
def save_model(model):
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d%H%M")
    file_name = "GCN_" + formatted_time + ".pth"
    file_path = os.path.join(r"result\GCN_model", file_name)

    torch.save(model.state_dict(), file_path)
    # print(f"Model saved to {file_path}")


# 5. 主训练和评估函数
@time_record
def GCN_train_and_evaluate(raw_data, truth_table, device, epochs=100, learning_rate=0.01):
    # 加载数据
    data = load_data(raw_data, truth_table, device)

    # 设置模型参数
    input_dim = data.x.size(1)  # 动态获取x的列数作为input_dim
    output_dim = len(set([community for _, community in truth_table]))  # 社区数量
    num_nodes = data.num_nodes  # 节点数

    # 根据数据规模动态设定 hidden_dim
    hidden_dim = max(16, int(math.sqrt(num_nodes) * input_dim * 0.5))  # 使用节点数和输入维度计算

    # 初始化模型
    model = GCN(input_dim, hidden_dim, output_dim).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 训练模型
    train(model, data, optimizer, loss_fn, epochs=epochs, device=device)

    # 获取最终社区划分结果
    ans = test(model, data, device=device)

    # 保存模型
    # save_model(model)

    return ans


# 示例运行
if __name__ == "__main__":
    # 示例数据
    raw_data = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]  # 边列表
    truth_table = [[0, 0], [1, 0], [2, 1], [3, 1]]  # 真实社区划分

    # 指定在 GPU 上运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 运行训练和评估
    communities = GCN_train_and_evaluate(raw_data, truth_table, device)
    print("社区划分结果:", communities)
