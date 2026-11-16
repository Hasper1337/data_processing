"""ЛР №2, задание 3, ВАРИАНТ 1: Y = 4*x1 - 2*x2.

Таблица 100 x 100 (x1, x2 от 1 до 100), 3D график, шум из [0.01; 0.1],
второй 3D график, преобразование в 3 столбца x1, x2, y и сохранение в csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # чтобы работало без окна; уберите, если хотите окно
import matplotlib.pyplot as plt

A, B = 4, -2                      # вариант 1: Y = 4*x1 - 2*x2
x1 = np.arange(1, 101)
x2 = np.arange(1, 101)
X1, X2 = np.meshgrid(x1, x2, indexing="ij")   # X1[i, j] = x1[i], X2[i, j] = x2[j]

# Таблица Y (как в Excel: строки - x1, столбцы - x2)
Y = A * X1 + B * X2
table = pd.DataFrame(Y, index=x1, columns=x2)
table.index.name = "x1\\x2"
table.to_csv("lab2_task3_table.csv")


def plot3d(Z, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="viridis")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Y")
    ax.set_title(title)
    fig.savefig(filename, dpi=120)
    plt.close(fig)


plot3d(Y, "Y = 4*x1 - 2*x2", "lab2_task3_plot_clean.png")

# Шум в интервале [0.01; 0.1]
rng = np.random.default_rng(0)
noise = rng.uniform(0.01, 0.1, size=Y.shape)
Y_noisy = Y + noise
plot3d(Y_noisy, "Y с шумом [0.01; 0.1]", "lab2_task3_plot_noisy.png")

# Три столбца: x1, x2, y
data = pd.DataFrame({"x1": X1.ravel(), "x2": X2.ravel(), "y": Y_noisy.ravel()})
data.to_csv("lab2_task3_data.csv", index=False)
print(data.head())
print("Строк:", len(data), "-> сохранено в lab2_task3_data.csv")
