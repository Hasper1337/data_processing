"""ЛР №2, задание 4: набор данных для бинарной классификации по цвету пикселя.

Нужна картинка pic.jpg (цветная). Выбираем один цвет TARGET_RGB, который есть на картинке.
Пиксели, близкие к этому цвету, получают метку 1, далёкие - метку 0.
Результат: lab2_task4_data.csv со столбцами R, G, B, label.

Запуск: python lab2_task4_color_dataset.py pic.jpg 255 255 0
(цвет можно не указывать - тогда возьмётся самый частый цвет на картинке)
"""
import sys

import numpy as np
import pandas as pd
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "pic.jpg"
img = np.array(Image.open(path).convert("RGB"))   # массив (высота, ширина, 3)
print("Картинка:", img.shape)

pixels = img.reshape(-1, 3).astype(int)

if len(sys.argv) >= 5:
    target = np.array([int(v) for v in sys.argv[2:5]])
else:
    # самый частый цвет (с округлением до 32, чтобы похожие цвета считались вместе)
    rounded = (pixels // 32) * 32 + 16
    values, counts = np.unique(rounded, axis=0, return_counts=True)
    target = values[counts.argmax()]
print("Выбранный цвет (R, G, B):", target)

dist = np.sqrt(((pixels - target) ** 2).sum(axis=1))   # расстояние до выбранного цвета
yes = pixels[dist < 40]      # цвет присутствует
no = pixels[dist > 120]      # цвета нет
print("Подходящих пикселей:", len(yes), "| неподходящих:", len(no))

N = min(500, len(yes), len(no))        # берём одинаковое число примеров каждого класса
rng = np.random.default_rng(0)
yes = yes[rng.choice(len(yes), N, replace=False)]
no = no[rng.choice(len(no), N, replace=False)]

data = pd.DataFrame(np.vstack([yes, no]), columns=["R", "G", "B"])
data["label"] = [1] * N + [0] * N
data = data.sample(frac=1, random_state=0).reset_index(drop=True)   # перемешать
data.to_csv("lab2_task4_data.csv", index=False)
print(data.head())
print("Сохранено:", len(data), "строк в lab2_task4_data.csv")
