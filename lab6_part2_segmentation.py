"""ЛР №6, часть 2: сегментация изображения кластеризацией пикселей.

Запуск: python lab6_part2_segmentation.py картинка1.jpg картинка2.png ...
(если картинки не указаны, берутся две тестовые картинки из sklearn: china.jpg и flower.jpg)

1) KMeans по признакам [lam*x, lam*y, r, g, b]  - lam задаёт вклад координат (пространственная связность);
2) спектральная кластеризация на уменьшенной картинке (она медленнее, поэтому картинка маленькая).
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import load_sample_images

MAX_SIDE_KMEANS = 200      # картинку уменьшаем, чтобы всё считалось быстро
MAX_SIDE_SPECTRAL = 50     # спектральная кластеризация - на совсем маленькой
N_CLUSTERS = 5


def load_images():
    if len(sys.argv) > 1:
        return [(path, Image.open(path).convert("RGB")) for path in sys.argv[1:]]
    sample = load_sample_images()
    return [(name.split("/")[-1], Image.fromarray(img)) for name, img in zip(sample.filenames, sample.images)]


def shrink(image, max_side):
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return np.asarray(image, dtype=float) / 255        # (h, w, 3), значения 0..1


def pixel_features(img, lam):
    """Признаки пикселя: [lam*x, lam*y, r, g, b], координаты нормированы на 0..1."""
    h, w, _ = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.stack([xs / w, ys / h], axis=-1) * lam
    return np.concatenate([coords, img], axis=-1).reshape(-1, 5)


def segment_kmeans(img, lam, k=N_CLUSTERS):
    labels = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(pixel_features(img, lam))
    return labels.reshape(img.shape[:2])


def paint_by_cluster(img, labels):
    """Каждый сегмент закрашиваем средним цветом его пикселей."""
    out = np.zeros_like(img)
    for c in np.unique(labels):
        out[labels == c] = img[labels == c].mean(axis=0)
    return out


for name, original in load_images():
    print("Картинка:", name, original.size)

    # ---- KMeans при разных lam ----
    img = shrink(original, MAX_SIDE_KMEANS)
    lams = [0, 0.5, 1, 2]
    fig, axes = plt.subplots(1, len(lams) + 1, figsize=(4 * (len(lams) + 1), 4))
    axes[0].imshow(img)
    axes[0].set_title("исходная")
    for ax, lam in zip(axes[1:], lams):
        segments = segment_kmeans(img, lam)
        ax.imshow(paint_by_cluster(img, segments))
        ax.set_title("KMeans, lam=%g" % lam)
    for ax in axes:
        ax.axis("off")
    fname = "lab6_kmeans_%s.png" % name.split(".")[0]
    plt.savefig(fname, dpi=100)
    plt.close()
    print("  KMeans:", fname, "(lam=0 - только цвет; чем больше lam, тем компактнее сегменты)")

    # ---- Спектральная кластеризация ----
    small = shrink(original, MAX_SIDE_SPECTRAL)
    spectral = SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors", n_neighbors=10,
                                  assign_labels="kmeans", random_state=0)
    spec_labels = spectral.fit_predict(pixel_features(small, 1)).reshape(small.shape[:2])
    kmeans_small = segment_kmeans(small, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(small)
    axes[0].set_title("исходная (уменьшенная)")
    axes[1].imshow(paint_by_cluster(small, kmeans_small))
    axes[1].set_title("KMeans")
    axes[2].imshow(paint_by_cluster(small, spec_labels))
    axes[2].set_title("спектральная")
    for ax in axes:
        ax.axis("off")
    fname = "lab6_spectral_%s.png" % name.split(".")[0]
    plt.savefig(fname, dpi=100)
    plt.close()
    print("  Спектральная кластеризация:", fname)

print("\nВывод: KMeans делит пиксели на выпуклые облака в пространстве признаков, спектральная")
print("кластеризация лучше выделяет связные области сложной формы, но работает намного медленнее")
print("(строит матрицу сходства пикселей), поэтому ей отдают маленькие изображения.")
