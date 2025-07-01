import numpy as np
import pandas as pd

from config import DATA_CONFIG


def load_data():
    df = pd.read_csv(DATA_CONFIG.DATA_PATH, sep='\t', names=DATA_CONFIG.DATA_COLUMNS)

    print(f"数据集加载完成: {len(df)}条评分记录")
    print(f"用户数量: {df['user_id'].nunique()}, 电影数量: {df['item_id'].nunique()}")

    return df

def build_rating_matrix(df):
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    # 创建从原始ID到矩阵索引的映射
    user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
    item_mapping = {item: idx for idx, item in enumerate(df['item_id'].unique())}

    # 构建评分矩阵
    R = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        user_idx = user_mapping[row['user_id']]
        item_idx = item_mapping[row['item_id']]
        R[user_idx, item_idx] = row['rating']

    sparsity = 100 * (1 - np.count_nonzero(R) / (n_users * n_items))

    print(f"评分矩阵构建完成 | 维度: {R.shape} | 稀疏度: {sparsity:.2f}%")

    return R, user_mapping, item_mapping


def load_movie_attributes():
    import pandas as pd
    import numpy as np
    from config import DATA_CONFIG

    # 读取 MovieLens u.item 数据
    movies = pd.read_csv(
        DATA_CONFIG.ITEM_PATH,
        sep="|",
        encoding="latin-1",
        header=None,
        names=DATA_CONFIG.ITEM_COLUMNS
    )

    # ===================== 类型属性（one-hot, 共19维） =====================
    genre_attrs = movies.iloc[:, 5:].astype(np.float32).values  # shape = (1682, 19)

    # ===================== 年份 scalar 属性（归一化） =====================
    release_years = []
    for date_str in movies["release_date"]:
        if pd.isna(date_str):
            year = 1900
        else:
            try:
                year = int(date_str.strip()[-4:])
            except:
                year = 1900
        release_years.append(year)

    years = np.array(release_years)
    min_year = years.min()
    max_year = years.max()

    year_scalars = ((years - min_year) / (max_year - min_year)).astype(np.float32)  # 归一化到 [0, 1]
    year_scalars = year_scalars.reshape(-1, 1)  # shape = (1682, 1)

    # ===================== 合并属性：类型 + 年份 =====================
    full_attrs = np.concatenate([genre_attrs, year_scalars], axis=1)  # shape = (1682, 20)

    # ===================== 构造 item_id → 属性映射 =====================
    attr_dict = {}
    for idx, row in movies.iterrows():
        item_id = int(row["movie_id"])
        attr_dict[item_id] = full_attrs[idx]

    num_attrs = full_attrs.shape[1]
    print(f"[INFO] 🎬 电影属性加载完成 | 类型属性: {genre_attrs.shape[1]} | 年份: scalar → 总维度: {num_attrs}")
    print(f"[INFO] 📆 年份归一化范围: [{min_year}, {max_year}] → [0.0, 1.0]")

    return attr_dict, num_attrs

def load_movie_attributes():
    # 读取带导演和演员的新文件
    movies = pd.read_csv(
        DATA_CONFIG.ITEM_WITH_DIRECTOR_AND_ACTOR_PATH,
        sep="|",
        encoding="latin-1",
        header=None
    )

    column_names = DATA_CONFIG.ITEM_COLUMNS + ['director', 'actors']
    movies.columns = column_names

    # ================== 类型属性（one-hot, 19维） ==================
    genre_attrs = movies.iloc[:, 5:24].astype(np.float32).values

    # ================== 年份属性 scalar ==================
    release_years = []
    for date_str in movies["release_date"]:
        if pd.isna(date_str):
            year = 1900
        else:
            try:
                year = int(date_str.strip()[-4:])
            except:
                year = 1900
        release_years.append(year)

    years = np.array(release_years)
    min_year, max_year = years.min(), years.max()
    year_scalars = ((years - min_year) / (max_year - min_year)).astype(np.float32).reshape(-1, 1)

    # ================== 导演 multi-hot 编码 ==================
    all_director_set = set()
    director_lists = []

    for d_str in movies["director"].fillna("Unknown"):
        directors = [d.strip() for d in str(d_str).split(",") if d.strip()]
        director_lists.append(directors)
        all_director_set.update(directors)

    unique_directors = sorted(all_director_set)
    director2idx = {d: i for i, d in enumerate(unique_directors)}
    director_multi_hot = np.zeros((len(movies), len(director2idx)), dtype=np.float32)
    for idx, directors in enumerate(director_lists):
        for d in directors:
            if d in director2idx:
                director_multi_hot[idx, director2idx[d]] = 1.0

    # ================== 演员 multi-hot 编码（取全部） ==================
    all_actor_set = set()
    actor_lists = []

    for actor_str in movies['actors'].fillna('Unknown'):
        actors = [a.strip() for a in str(actor_str).split(',') if a.strip()]
        actor_lists.append(actors)
        all_actor_set.update(actors)

    unique_actors = sorted(all_actor_set)
    actor2idx = {a: i for i, a in enumerate(unique_actors)}
    actor_multi_hot = np.zeros((len(movies), len(actor2idx)), dtype=np.float32)
    for idx, actors in enumerate(actor_lists):
        for a in actors:
            if a in actor2idx:
                actor_multi_hot[idx, actor2idx[a]] = 1.0

    # ================== 合并所有属性 ==================
    full_attrs = np.concatenate([genre_attrs, year_scalars, director_multi_hot, actor_multi_hot], axis=1)

    print(f"[INFO] 🎬 属性维度统计:")
    print(f" - 类型维度: {genre_attrs.shape[1]}")
    print(f" - 年份维度: {year_scalars.shape[1]}")
    print(f" - 导演维度: {director_multi_hot.shape[1]}")
    print(f" - 演员维度: {actor_multi_hot.shape[1]}")
    print(f" - 合并后维度: {full_attrs.shape[1]}")

    # 构造 item_id → 属性向量
    attr_dict = {}
    for idx, row in movies.iterrows():
        item_id = int(row["movie_id"])
        attr_dict[item_id] = full_attrs[idx]

    return attr_dict, full_attrs.shape[1]
