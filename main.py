import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import MMF_CONFIG
from data.loader import load_data, build_rating_matrix, load_movie_attributes
from model.mf import MatrixFactorization
from model.mmf import MMF
from model.visualizer import MMFVisualizer


def calculate_rmse(model, test_df, user_mapping, item_mapping):
    """计算测试集的RMSE"""
    y_true = []
    y_pred = []

    for idx, row in test_df.iterrows():
        user_id = row['user_id']  # 根据实际列名调整
        item_id = row['item_id']  # 根据实际列名调整
        rating = row['rating']  # 根据实际列名调整

        # 获取映射后的索引
        u_idx = user_mapping.get(user_id)
        i_idx = item_mapping.get(item_id)

        # 只处理训练集中存在的用户和物品
        if u_idx is not None and i_idx is not None:
            pred = model.predict(u_idx, i_idx)
            # print("y_true:",rating, "y_pred:",pred)

            y_true.append(rating)
            y_pred.append(pred)

    # 计算RMSE
    mse = mean_squared_error(y_true, y_pred)
    print()
    return np.sqrt(mse)


def mf_result(R_train, test_df, user_mapping, item_mapping):
    # 初始化模型
    model = MatrixFactorization(R_train)

    # 训练模型
    print("\n开始训练矩阵分解模型...")
    model.train()

    # 计算测试集RMSE
    rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
    print(f"\n测试集RMSE: {rmse:.4f}")


def mmf_result(train_df, test_df):
    # Step 1: 读取所有 item 属性
    all_attrs_dict, num_attrs = load_movie_attributes()  # shape = (1682, 19)

    # Step 2: 创建 user 和 item 映射（仅限训练集）
    user_mapping = {id: idx for idx, id in enumerate(train_df["user_id"].unique())}
    item_mapping = {id: idx for idx, id in enumerate(train_df["item_id"].unique())}

    # Step 3: 构建训练评分矩阵 R_train
    R_train = np.zeros((len(user_mapping), len(item_mapping)))
    for _, row in train_df.iterrows():
        u = user_mapping[row["user_id"]]
        i = item_mapping[row["item_id"]]
        R_train[u, i] = row["rating"]

    # Step 4: 对 item_attrs 做映射，过滤训练集中存在的 item
    item_attrs_aligned = np.zeros((len(item_mapping), num_attrs), dtype=np.float32)

    for raw_id, mapped_idx in item_mapping.items():
        item_attrs_aligned[mapped_idx] = all_attrs_dict.get(raw_id, np.zeros(num_attrs)) # MovieLens item_id 从 1 开始

    # Step 5: 训练模型
    if MMF_CONFIG.USE_CACHE:
        model = MMF.load(MMF_CONFIG.MODEL_PATH)
    else:
        model = MMF(R_train, item_attrs_aligned, k=MMF_CONFIG.LATENT_K, alpha=MMF_CONFIG.ALPHA, beta=MMF_CONFIG.BETA,
                    epochs=MMF_CONFIG.EPOCHS)
        model.train()

     # Step 6: 测试评估
    rmse = calculate_rmse(model, test_df, user_mapping, item_mapping)
    print(f"\nmmf测试集RMSE: {rmse:.4f}")

    os.makedirs(f"saved_models/rmse_{rmse:.4f}", exist_ok=True)
    model.save(MMF_CONFIG.MODEL_PATH)

    # 可视化一条预测
    example_row = test_df.iloc[0]
    user_id = example_row["user_id"]
    item_id = example_row["item_id"]
    u_idx = user_mapping.get(user_id)
    i_idx = item_mapping.get(item_id)

    genre_names = [
        "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    visualizer = MMFVisualizer(model, genre_names=genre_names)
    visualizer.log_prediction_details(u_idx, i_idx)
    #
    # # 选取测试集中前几条
    # for idx in range(3):
    #     row = test_df.iloc[idx]
    #     user_id = row["user_id"]
    #     item_id = row["item_id"]
    #     rating = row["rating"]
    #
    #     u_idx = user_mapping.get(user_id)
    #     i_idx = item_mapping.get(item_id)
    #
    #     print(f"\n[🔍 Case {idx + 1}]")
    #     print(f"user_id: {user_id}, mapped index: {u_idx}")
    #     print(f"item_id: {item_id}, mapped index: {i_idx}")
    #     print(f"真实评分 rating: {rating}")
    #
    #     # 属性向量对齐验证
    #     all_attrs_dict, _ = load_movie_attributes()
    #
    #     true_attr = all_attrs_dict[item_id]
    #     aligned_attr = model.item_attrs[i_idx]
    #
    #     print(f"🎯 对齐验证: 属性是否一致 → {np.allclose(true_attr, aligned_attr)}")
    #     print(f"原始属性:    {true_attr}")
    #     print(f"对齐后属性:  {aligned_attr}")
    #
    #     # 进一步打印 θ、ω、U、F 维度内容
    #     genres = np.where(aligned_attr > 0)[0]
    #     for g in genres:
    #         dot_val = np.dot(model.U[u_idx], model.F[g])
    #         theta = model.theta[i_idx, g]
    #         omega = model.omega[u_idx, g]
    #         contrib = omega * theta * dot_val
    #         print(f"属性: {g} | θ={theta:.4f}, ω={omega:.4f}, dot={dot_val:.4f} → 贡献: {contrib:.4f}")


if __name__ == "__main__":
    # 加载并预处理数据
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 构建评分矩阵
    R_train, user_mapping, item_mapping = build_rating_matrix(train_df)

    # mf 矩阵
    # mf_result(R_train, test_df, user_mapping, item_mapping)

    # mmf 矩阵
    mmf_result(train_df, test_df)


