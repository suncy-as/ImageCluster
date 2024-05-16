# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import numpy as np
from sklearn.mixture import GaussianMixture

def image_clustering_3D(image_path, n_clusters):
    # 读取3D图像数据
    image = nib.load(image_path)
    image_data = image.get_fdata()

    # 对3D数据进行高斯混合模型聚类
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(image_data.reshape(-1, 1))
    labels = gmm.predict(image_data.reshape(-1, 1))

    # 将聚类结果映射回原始3D形状
    clustered_image = labels.reshape(image_data.shape)

    return clustered_image

