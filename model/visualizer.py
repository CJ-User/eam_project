import numpy as np
import matplotlib.pyplot as plt


class MMFVisualizer:
    def __init__(self, model, genre_names=None):
        self.model = model
        self.item_attrs = model.item_attrs
        self.genre_names = genre_names or [f"Genre_{i}" for i in range(self.item_attrs.shape[1])]

    def log_prediction_details(self, u_idx, i_idx):
        pass
        # genres = np.where(self.item_attrs[i_idx] > 0)[0]
        # u_vec = self.model.U[u_idx]
        #
        # print("\n======================")
        # print(f"🧪 分析用户 {u_idx} 对电影 {i_idx} 的评分预测")
        #
        # total_contrib = 0.0
        # for g in genres:
        #     f_g = self.model.F[g]
        #     dot = np.dot(u_vec, f_g)
        #     theta = self.model.theta[i_idx, g]
        #     omega = self.model.omega[u_idx, g]
        #     contrib = omega * theta * dot
        #     total_contrib += contrib
        #
        #     print(f"\n属性: {self.genre_names[g]}")
        #     print(f"- ω[{u_idx}, {g}] = {omega:.4f}")
        #     print(f"- θ[{i_idx}, {g}] = {theta:.4f}")
        #     print(f"- u^T f[{g}] = {dot:.6f}")
        #     print(f"→ 属性评分贡献: ω × θ × (u^T f) = {contrib:.6f}")
        #
        # pred_rating = total_contrib / (len(genres) + 1e-8)
        # print(f"\nΣ 属性贡献: {total_contrib:.6f}")
        # print(f"预测评分: 属性贡献平均 = {pred_rating:.4f}")
        # print("======================\n")
