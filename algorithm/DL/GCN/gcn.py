import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import math
from sklearn.metrics import normalized_mutual_info_score
import networkx as nx
from networkx.algorithms.community.quality import modularity
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from common.util.decorator import time_record


# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 数据加载和预处理
def load_data(raw_data, truth_table, device, train_ratio=0.8):
    nodes = set([x for edge in raw_data for x in edge])
    num_nodes = max(nodes) + 1
    edge_index = torch.tensor(raw_data, dtype=torch.long).t().contiguous().to(device)
    y = torch.zeros(num_nodes, dtype=torch.long).to(device)
    for node, community in truth_table:
        y[node] = community
    num_truth = len(truth_table)
    indices = torch.randperm(num_truth)
    train_size = int(train_ratio * num_truth)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    degrees = degree(edge_index[0], num_nodes=num_nodes).view(-1, 1).to(device)
    random_features = torch.randn(num_nodes, 16, dtype=torch.float).to(device)
    identity_features = torch.eye(num_nodes, dtype=torch.float).to(device)
    x = torch.cat([degrees, random_features, identity_features], dim=1)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    data.train_mask, data.test_mask = train_mask, test_mask
    return data

# 训练函数
def train(model, data, optimizer, loss_fn, epochs=100, device='cpu'):
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()  # 强制同步，确保所有 GPU 操作完成

# 测试函数
def test(model, data, device='cpu'):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        predictions = output.argmax(dim=1)
    correct = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = correct.item() / data.test_mask.sum().item()
    accuracy = round(accuracy, 16)
    nmi = round(normalized_mutual_info_score(data.y[data.test_mask].cpu(), predictions[data.test_mask].cpu()), 16)
    edge_list = data.edge_index.cpu().t().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    communities = {label: [] for label in set(predictions.cpu().numpy())}
    for idx, community in enumerate(predictions.cpu().numpy()):
        communities[community].append(idx)
    mod = round(modularity(G, list(communities.values())), 16)
    return accuracy, nmi, mod

# 主训练和评估函数
@time_record
def GCN_train_and_evaluate(raw_data, truth_table, device, epochs=100, learning_rate=0.01, batch_size=32):
    data = load_data(raw_data, truth_table, device)
    input_dim = data.x.size(1)
    output_dim = len(set([community for _, community in truth_table]))
    num_nodes = data.num_nodes
    hidden_dim = min(max(16, int(math.sqrt(num_nodes) * input_dim * 0.5)), 65536)
    model = GCN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 增加批次处理数据以更好利用 GPU 资源
    loader = DataLoader([data], batch_size=batch_size, shuffle=True)

    # 记录训练开始时间
    start_time = time.time()

    # 训练模型
    for batch_data in loader:
        train(model, batch_data, optimizer, loss_fn, epochs=epochs, device=device)

    # 记录结束时间
    runtime = time.time() - start_time

    # 获取测试集上的准确率、NMI 和 Modularity
    accuracy, nmi, mod = test(model, data, device=device)
    return accuracy, nmi, mod, runtime

# 示例运行
if __name__ == "__main__":
    raw_data = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]  # 边列表
    truth_table = [[0, 0], [1, 0], [2, 1], [3, 1]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 每次运行清空 GPU 缓存，减少碎片
    torch.cuda.empty_cache()
    accuracy, nmi, mod, runtime = GCN_train_and_evaluate(raw_data, truth_table, device)
    print("Test Accuracy:", accuracy)
    print("Test NMI:", nmi)
    print("Modularity:", mod)
    print("Training Runtime:", runtime, "seconds")
