import os
import pickle

import numpy as np

from config import MMF_CONFIG


class MMF:
    def __init__(self, R, item_attrs, k=40, alpha=0.02, beta=0.005, epochs=150):
        self.R = R
        self.item_attrs = item_attrs
        self.num_users, self.num_items = R.shape
        self.num_genres = item_attrs.shape[1]
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

        self.U = np.random.normal(scale=1. / k, size=(self.num_users, k))
        self.F = np.random.normal(scale=1. / k, size=(self.num_genres, k))

        self.omega = np.zeros((self.num_users, self.num_genres), dtype=np.float32)
        for g in range(self.num_genres):
            if np.any(item_attrs[:, g] > 0):
                # self.omega[:, g] = np.random.uniform(0.1, 1.5, size=self.num_users)
                self.omega[:, g] = np.random.uniform(0.5, 2.0, size=self.num_users)


        self.theta = np.zeros_like(item_attrs, dtype=np.float32)
        non_zero_mask = item_attrs > 0
        # self.theta[non_zero_mask] = np.random.uniform(0.05, 0.15, size=np.sum(non_zero_mask))
        self.theta[non_zero_mask] = np.random.uniform(0.1, 0.5, size=np.sum(non_zero_mask))

        self.user_indices, self.item_indices = np.where(R > 0)
        self.samples = list(zip(self.user_indices, self.item_indices))

        # 取一个样本点初始预测值
        u, i = self.samples[0]
        init_pred = self.predict(u, i)
        print(f"[Init Predict Debug] u={u}, i={i}, predicted rating={init_pred:.4f}")

    def predict(self, u, i):
        genres = np.where(self.item_attrs[i] > 0)[0]
        total = 0.0
        for g in genres:
            dot_val = np.dot(self.U[u], self.F[g])
            dot_val = np.clip(dot_val, -5, 5)
            total += self.omega[u, g] * self.theta[i, g] * dot_val

        pred = total / len(genres)
        return np.clip(pred, 1, 5)

    def train(self, verbose=True):
        # Step 3: 打印初始预测值用于 sanity check
        u = np.random.randint(0, self.num_users)
        i = np.random.randint(0, self.num_items)
        initial_pred = self.predict(u, i)
        print(f"[Init Predict Debug] u={u}, i={i}, predicted rating={initial_pred:.4f}")

        dot_vals = []
        for epoch in range(self.epochs):
            current_alpha = self.alpha * (0.95 ** (epoch // 5))
            np.random.shuffle(self.samples)

            total_loss = 0.0

            for idx, (u, i) in enumerate(self.samples):
                r_ui = self.R[u, i]
                r_hat = self.predict(u, i)
                e_ui = r_ui - r_hat
                total_loss += e_ui ** 2

                items = np.where(self.item_attrs[i] > 0)[0]
                for item in items:
                    # dot = np.dot(self.U[u], self.F[item])
                    # # dot = np.clip(dot, -10, 10)
                    # dot = np.clip(dot, -5, 5)
                    #
                    # dot_vals.append(dot)
                    #
                    # grad_U = e_ui * self.omega[u, item] * self.theta[i, item] * self.F[item]
                    # grad_F = e_ui * self.omega[u, item] * self.theta[i, item] * self.U[u]
                    # grad_omega = e_ui * self.theta[i, item] * dot
                    # grad_theta = e_ui * self.omega[u, item] * dot

                    normalization = len(items)

                    dot = np.dot(self.U[u], self.F[item])
                    # dot = np.clip(dot, -5, 5)
                    dot_vals.append(dot)

                    grad_U = (e_ui * self.omega[u, item] * self.theta[i, item] * self.F[item]) / normalization
                    grad_F = (e_ui * self.omega[u, item] * self.theta[i, item] * self.U[u]) / normalization
                    grad_omega = (e_ui * self.theta[i, item] * dot) / normalization
                    grad_theta = (e_ui * self.omega[u, item] * dot) / normalization

                    for grad in [grad_U, grad_F]:
                        norm = np.linalg.norm(grad)
                        if norm > 1.0:
                            grad /= norm

                    # 在梯度更新前添加
                    # grad_U = np.clip(grad_U, -0.5, 0.5)
                    # grad_F = np.clip(grad_F, -0.5, 0.5)

                    self.U[u] += current_alpha * (grad_U - self.beta * self.U[u])
                    self.F[item] += current_alpha * (grad_F - self.beta * self.F[item])

                    new_omega = self.omega[u, item] + current_alpha * (grad_omega - self.beta * self.omega[u, item])
                    new_theta = self.theta[i, item] + current_alpha * (grad_theta - self.beta * self.theta[i, item])
                    self.omega[u, item] = np.clip(new_omega, MMF_CONFIG.OMEGA_CLIP[0], MMF_CONFIG.OMEGA_CLIP[1])
                    self.theta[i, item] = np.clip(new_theta, MMF_CONFIG.THETA_CLIP[0], MMF_CONFIG.THETA_CLIP[1])

            avg_loss = total_loss / len(self.samples)
            rmse = np.sqrt(avg_loss)

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                dot_min = np.min(dot_vals)
                dot_max = np.max(dot_vals)
                dot_mean = np.mean(dot_vals)
                dot_std = np.std(dot_vals)

                print(
                    f"[Epoch {epoch + 1}/{self.epochs}] RMSE: {rmse:.4f} | AvgLoss: {avg_loss:.4f} | LR: {current_alpha:.5f}")
                print(
                    f"[θ] mean: {np.mean(self.theta):.4f}, std: {np.std(self.theta):.4f}, max: {np.max(self.theta):.4f}")
                print(
                    f"[ω] mean: {np.mean(self.omega):.4f}, std: {np.std(self.omega):.4f}, max: {np.max(self.omega):.4f}")
                print(f"[dot(U,F)] mean={dot_mean:.4e}, std={dot_std:.4e}, range=({dot_min:.4f}, {dot_max:.4f})")

                u_debug, i_debug = self.samples[10]
                pred_debug = self.predict(u_debug, i_debug)
                print(f"[Debug Predict@Epoch {epoch + 1}] u={u_debug}, i={i_debug}, pred={pred_debug:.4f}")

        print("\n[Final Params Summary]")
        print(f"U mean: {np.mean(self.U):.4f}, F mean: {np.mean(self.F):.4f}")
        print(f"theta mean: {np.mean(self.theta):.4f}, omega mean: {np.mean(self.omega):.4f}")

    def evaluate_rmse(self):
        mse = 0.0
        for u, i in self.samples:
            r_ui = self.R[u, i]
            r_hat = self.predict(u, i)
            mse += (r_ui - r_hat) ** 2
        return np.sqrt(mse / len(self.samples))

    def save(self, file_path):
        """保存模型到指定路径"""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] 模型已保存到 {file_path}")

    @staticmethod
    def load(file_path):
        """从文件加载模型"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] 模型文件未找到: {file_path}")
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] 模型已从 {file_path} 加载")
        return model
