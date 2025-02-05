import math

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


# 定义图自编码器模型
class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # 返回嵌入表示


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim)

    def forward(self, data):
        z = self.encoder(data)  # 得到节点嵌入
        return z


# 数据加载和预处理
def load_data(raw_data, device):
    nodes = set([x for edge in raw_data for x in edge])
    num_nodes = max(nodes) + 1

    edge_index = torch.tensor(raw_data, dtype=torch.long).t().contiguous().to(device)
    degrees = degree(edge_index[0], num_nodes=num_nodes)
    degrees = degrees.view(-1, 1).to(device)  # 将度数转为列向量形状

    random_features = torch.randn(num_nodes, 16, dtype=torch.float).to(device)
    identity_features = torch.eye(num_nodes, dtype=torch.float).to(device)

    x = torch.cat([degrees, random_features, identity_features], dim=1)

    data = Data(x=x, edge_index=edge_index).to(device)
    return data


# 对比损失函数
def contrastive_loss(z, edge_index, margin=1.0):
    pos_i, pos_j = edge_index
    pos_loss = F.pairwise_distance(z[pos_i], z[pos_j]).pow(2).mean()

    neg_i = pos_i
    neg_j = torch.randint(0, z.size(0), pos_i.size(), device=z.device)
    neg_loss = F.relu(margin - F.pairwise_distance(z[neg_i], z[neg_j])).pow(2).mean()

    return pos_loss + neg_loss


# 训练函数，增加提前停止机制
def train_autoencoder(
    model,
    data,
    optimizer,
    scheduler,
    epochs=100,
    margin=1.0,
    patience=10,
    tolerance=1e-4,
):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data)  # 获取嵌入

        # 计算对比损失
        loss = contrastive_loss(z, data.edge_index, margin=margin)
        loss.backward()
        optimizer.step()

        # 学习率调度器，根据损失调整学习率
        scheduler.step(loss)

        # 检查损失是否有显著改善
        if loss < best_loss - tolerance:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # 提前停止条件
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} with loss {loss.item():.4f}")
            break

        # 打印损失和学习率信息
        if (epoch + 1) % 100 == 0 or patience_counter >= patience:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}"
            )

    return best_loss.item()


# 使用 KMeans 对节点嵌入进行聚类以得到社区划分
def cluster_communities(embeddings, num_communities=2):
    kmeans = KMeans(n_clusters=num_communities)
    labels = kmeans.fit_predict(embeddings.cpu().detach().numpy())

    communities = {}
    for node, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node)

    return list(communities.values())


# 主函数，进行无监督训练
def GCN_train_unsupervised(
    raw_data,
    device,
    epochs=100,
    learning_rate=0.01,
    margin=1.0,
    patience=100,
    tolerance=1e-4,
):
    data = load_data(raw_data, device)

    input_dim = data.x.size(1)
    num_nodes = data.num_nodes
    hidden_dim = min(65536, max(16, int(math.sqrt(num_nodes) * input_dim * 0.5)))
    embedding_dim = 64

    model = GraphAutoencoder(input_dim, hidden_dim, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 使用 ReduceLROnPlateau 调度器，根据验证损失动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_autoencoder(
        model,
        data,
        optimizer,
        scheduler,
        epochs=epochs,
        margin=margin,
        patience=patience,
        tolerance=tolerance,
    )

    embeddings = model.encoder(data)  # 获取节点嵌入
    communities = cluster_communities(embeddings, num_communities=2)  # 对嵌入进行聚类

    return communities


# 示例运行
if __name__ == "__main__":
    raw_data = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    communities = GCN_train_unsupervised(
        raw_data,
        device,
        epochs=1000,
        learning_rate=0.01,
        margin=1.0,
        patience=20,
        tolerance=1e-4,
    )
    print("社区划分结果:", communities)
