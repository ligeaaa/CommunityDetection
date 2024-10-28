import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity
import networkx as nx

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

# 封装接口函数
def run_cora_classification(content_path, cites_path, epochs=200, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定在 GPU 上运行

    # 加载 Cora 数据集并构建图结构
    def load_cora_data(content_path, cites_path):
        with open(content_path, 'r') as f:
            content = f.readlines()
        node_features = []
        node_labels = []
        node_ids = []
        label_dict = {}
        label_count = 0

        for line in content:
            items = line.strip().split()
            node_id = int(items[0])
            features = list(map(int, items[1:-1]))
            label = items[-1]
            if label not in label_dict:
                label_dict[label] = label_count
                label_count += 1
            label_id = label_dict[label]

            node_ids.append(node_id)
            node_features.append(features)
            node_labels.append(label_id)

        node_features = torch.tensor(node_features, dtype=torch.float).to(device)
        node_labels = torch.tensor(node_labels, dtype=torch.long).to(device)

        edge_index = []
        with open(cites_path, 'r') as f:
            for line in f.readlines():
                src, dest = map(int, line.strip().split())
                src_idx = node_ids.index(src)
                dest_idx = node_ids.index(dest)
                edge_index.append([src_idx, dest_idx])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

        return node_features, node_labels, edge_index, len(label_dict)

    # 加载数据
    node_features, node_labels, edge_index, num_classes = load_cora_data(content_path, cites_path)
    data = Data(x=node_features, edge_index=edge_index, y=node_labels).to(device)

    # 创建数据掩码
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_ratio = 0.8
    train_size = int(train_ratio * num_nodes)
    data.train_mask[:train_size] = True
    data.test_mask[train_size:] = True

    # 初始化模型并将其移动到指定设备
    model = GCN(input_dim=node_features.shape[1], hidden_dim=64, output_dim=num_classes).to(device)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    # 记录结束时间
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Training Runtime: {runtime:.4f} seconds")

    # 测试模型
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    # 计算 Accuracy，精确到小数点后16位
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = round(int(correct) / int(data.test_mask.sum()), 16)
    print(f'Test Accuracy: {acc:.16f}')

    # 计算 NMI，精确到小数点后16位
    nmi = round(normalized_mutual_info_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu()), 16)
    print(f'Test NMI: {nmi:.16f}')

    # 计算 Modularity，精确到小数点后16位
    edge_list = edge_index.cpu().t().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_list)
    communities = {label: [] for label in set(pred.cpu().numpy())}
    for idx, community in enumerate(pred.cpu().numpy()):
        communities[community].append(idx)
    communities = list(communities.values())
    mod = round(modularity(G, communities), 16)
    print(f'Modularity: {mod:.16f}')

    return model, runtime, acc, nmi, mod


if __name__ == '__main__':
    # 使用接口示例
    content_path = "path_to_cora.content"  # 请将此路径替换为您的 cora.content 文件路径
    cites_path = "path_to_cora.cites"  # 请将此路径替换为您的 cora.cites 文件路径

    model, runtime, accuracy, nmi, modularity_value = run_cora_classification(content_path, cites_path)
    print("最终测试准确率:", accuracy)
    print("NMI:", nmi)
    print("Modularity:", modularity_value)
