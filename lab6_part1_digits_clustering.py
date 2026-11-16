"""ЛР №6, часть 1: кластеризация рукописных цифр (sklearn.datasets.load_digits).

KMeans с разной инициализацией, выбор числа кластеров, метрики качества,
центры кластеров как картинки, PCA и t-SNE, ошибочно кластеризованные объекты.
Графики сохраняются в PNG.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, completeness_score, homogeneity_score,
                             silhouette_score)

digits = load_digits()
X, y = digits.data, digits.target
print("Данные:", X.shape, "| классов:", len(np.unique(y)))


def report(name, labels):
    """Метрики качества кластеризации."""
    print("%-34s однородность=%.3f полнота=%.3f ARI=%.3f силуэт=%.3f" % (
        name, homogeneity_score(y, labels), completeness_score(y, labels),
        adjusted_rand_score(y, labels), silhouette_score(X, labels)))


# ---------------------------------------------------------------------
# 1. KMeans с разной инициализацией (10 кластеров = 10 цифр)
# ---------------------------------------------------------------------
print("\n1. Разные способы инициализации (n_init=1, один запуск):")
for init in ["random", "k-means++"]:
    km = KMeans(n_clusters=10, init=init, n_init=1, random_state=0).fit(X)
    report("init=%s" % init, km.labels_)

# запуск с несколькими инициализациями выбирает лучший по inertia
km = KMeans(n_clusters=10, init="k-means++", n_init=10, random_state=0).fit(X)
report("k-means++, n_init=10", km.labels_)

# инициализация центрами - по одному объекту каждой цифры (знаем метки - для сравнения)
init_centers = np.array([X[y == d][0] for d in range(10)])
km_fixed = KMeans(n_clusters=10, init=init_centers, n_init=1).fit(X)
report("init = по объекту каждой цифры", km_fixed.labels_)

# ---------------------------------------------------------------------
# 2. Выбор числа кластеров: инерция ("локоть") и силуэт
# ---------------------------------------------------------------------
ks = list(range(2, 21))
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, model.labels_))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ks, inertias, "o-")
axes[0].set_title("Инерция (метод локтя)")
axes[0].set_xlabel("число кластеров")
axes[1].plot(ks, silhouettes, "o-")
axes[1].set_title("Силуэт")
axes[1].set_xlabel("число кластеров")
plt.savefig("lab6_choose_k.png", dpi=110)
plt.close()
print("\n2. Лучший силуэт при k =", ks[int(np.argmax(silhouettes))], "(график lab6_choose_k.png)")
print("   Истинное число классов - 10; метрики без меток (инерция, силуэт) дают лишь ориентир.")

# ---------------------------------------------------------------------
# 3. Центры кластеров как картинки 8x8
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, center, number in zip(axes.ravel(), km.cluster_centers_, range(10)):
    ax.imshow(center.reshape(8, 8), cmap="gray_r")
    ax.set_title("кластер %d" % number)
    ax.axis("off")
plt.savefig("lab6_centroids.png", dpi=110)
plt.close()
print("3. Центры кластеров: lab6_centroids.png")

# ---------------------------------------------------------------------
# 4. Какой цифре соответствует каждый кластер и какие объекты ошибочные
# ---------------------------------------------------------------------
cluster_to_digit = {}
for c in range(10):
    cluster_to_digit[c] = int(np.bincount(y[km.labels_ == c], minlength=10).argmax())   # самая частая цифра
predicted = np.array([cluster_to_digit[c] for c in km.labels_])
wrong = np.where(predicted != y)[0]
print("4. Соответствие кластер -> цифра:", cluster_to_digit)
print("   Ошибочно кластеризовано: %d из %d (%.1f%%)" % (len(wrong), len(y), 100 * len(wrong) / len(y)))

fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, i in zip(axes.ravel(), wrong[:16]):
    ax.imshow(X[i].reshape(8, 8), cmap="gray_r")
    ax.set_title("цифра %d -> %d" % (y[i], predicted[i]), fontsize=8)
    ax.axis("off")
plt.savefig("lab6_wrong.png", dpi=110)
plt.close()
print("   Примеры ошибок: lab6_wrong.png (смотрите, какие цифры путаются - обычно похожие по форме)")

# ---------------------------------------------------------------------
# 5. Снижение размерности: PCA и t-SNE (2 компоненты - для рисунка)
# ---------------------------------------------------------------------
X_pca2 = PCA(n_components=2, random_state=0).fit_transform(X)
X_tsne2 = TSNE(n_components=2, random_state=0).fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, emb, title in zip(axes, [X_pca2, X_tsne2], ["PCA, 2 компоненты", "t-SNE, 2 компоненты"]):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8)
    ax.set_title(title)
fig.colorbar(sc, ax=axes, label="цифра")
plt.savefig("lab6_pca_tsne.png", dpi=110)
plt.close()
print("\n5. Проекции: lab6_pca_tsne.png (t-SNE разделяет классы заметно лучше PCA)")

# ---------------------------------------------------------------------
# 6. KMeans после снижения размерности
# ---------------------------------------------------------------------
print("\n6. KMeans на данных после снижения размерности:")
X_pca10 = PCA(n_components=10, random_state=0).fit_transform(X)
for name, data in [("PCA, 2 компоненты", X_pca2), ("PCA, 10 компонент", X_pca10),
                   ("t-SNE, 2 компоненты", X_tsne2)]:
    labels = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(data)
    # метрики с метками считаем по меткам, силуэт - по исходным данным
    print("%-22s однородность=%.3f полнота=%.3f ARI=%.3f" % (
        name, homogeneity_score(y, labels), completeness_score(y, labels),
        adjusted_rand_score(y, labels)))
print("   Для сравнения исходные 64 признака: ARI = %.3f" % adjusted_rand_score(y, km.labels_))
