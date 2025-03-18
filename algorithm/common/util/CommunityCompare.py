import numpy as np
from scipy.optimize import linear_sum_assignment


class CommunityComparator:
    def __init__(self, computed_communities, truth_table):
        self.computed_communities = computed_communities
        self.truth_table = truth_table
        self.assignment = None
        self.metrics = {}

    def compare_sizes(self):
        """检查社区数量是否一致"""
        return len(self.computed_communities) == len(self.truth_table)

    def compute_assignment(self):
        """使用匈牙利算法匹配计算的社区和真实社区"""
        num_computed = len(self.computed_communities)
        num_truth = len(self.truth_table)
        n = max(num_computed, num_truth)
        cost_matrix = np.zeros((n, n))

        for i in range(num_truth):
            for j in range(num_computed):
                intersection = len(
                    set(self.truth_table[i]) & set(self.computed_communities[j])
                )
                cost_matrix[i, j] = -intersection  # 负值使得最大匹配变成最小成本问题

        row_ind, col_ind = linear_sum_assignment(cost_matrix[:num_truth, :num_computed])
        self.assignment = list(zip(row_ind, col_ind))

    def compute_metrics(self):
        """计算每个真实社区的 TP, FN, FP，并计算精确度和召回率"""
        assigned_truth = {i: j for i, j in self.assignment}  # 真实社区 -> 计算社区

        for i, true_comm in enumerate(self.truth_table):
            computed_index = assigned_truth.get(i, None)
            if computed_index is not None and computed_index < len(
                self.computed_communities
            ):
                comp_comm = self.computed_communities[computed_index]
                tp_set = set(comp_comm) & set(true_comm)
                fn_set = set(true_comm) - tp_set
                fp_set = set(comp_comm) - tp_set
                precision = (
                    len(tp_set) / (len(tp_set) + len(fp_set))
                    if (len(tp_set) + len(fp_set)) > 0
                    else 0
                )
                recall = (
                    len(tp_set) / (len(tp_set) + len(fn_set))
                    if (len(tp_set) + len(fn_set)) > 0
                    else 0
                )
                self.metrics[i] = {
                    "tp": len(tp_set),
                    "fn": len(fn_set),
                    "fp": len(fp_set),
                    "precision": precision,
                    "recall": recall,
                }
            else:
                self.metrics[i] = {
                    "tp": 0,
                    "fn": len(true_comm),
                    "fp": 0,
                    "precision": 0,
                    "recall": 0,
                }

    def print_results(self):
        """打印匹配结果和计算的指标，按照社区大小排序"""
        print(f"Truth_table 中社区数量: {len(self.truth_table)}")
        print(f"实验结果中社区数量: {len(self.computed_communities)}")
        print("\nTruth_table  <- 实验结果")
        for truth, computed in self.assignment:
            print(f"{truth + 1}  <- {computed + 1}")
        print("\n详细匹配结果 (按社区大小排序):")
        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: len(self.truth_table[x[0]]),
            reverse=True,
        )
        for comm_id, values in sorted_metrics:
            print(
                f"社区 {comm_id + 1}: TP={values['tp']}, FN={values['fn']}, FP={values['fp']}, 精确度={values['precision']:.4f}, 召回率={values['recall']:.4f}"
            )

    def run(self):
        """执行完整的社区比较流程"""
        self.compute_assignment()
        self.compute_metrics()
        self.print_results()
        return {"assignment": self.assignment, "metrics": self.metrics}
