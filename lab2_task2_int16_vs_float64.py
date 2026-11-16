"""ЛР №2, задание 2: что быстрее - float64 или int16? (pandas и numpy)

Данные: city_temperature.csv из 2data.zip (Kaggle, daily temperature of major cities).
Положите файл рядом со скриптом или передайте путь: python lab2_task2_int16_vs_float64.py путь.csv

Замечание: в тексте задания написано, что float64 занимает 4 байта, на самом
деле float64 - это 8 байт, а int16 - 2 байта.
"""
import sys
import datetime

import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "city_temperature.csv"
REPEAT = 200  # число повторов, чтобы время было заметным

# --- Pandas ---
df = pd.read_csv(path, low_memory=False)
print(df.head())
print(df.shape)
print(df.dtypes)

# Сколько записей имеют нулевое значение (пропуск) по полю State?
print("Пропусков в State:", df["State"].isna().sum())

# Новый столбец: температура * 10, приведённая к int
df["TempInt"] = (df["AvgTemperature"] * 10).astype(int)


def measure(func):
    """Возвращает время выполнения func() REPEAT раз (секунды)."""
    start = datetime.datetime.now()
    for _ in range(REPEAT):
        func()
    return (datetime.datetime.now() - start).total_seconds()


t_float = measure(df["AvgTemperature"].mean)
t_int = measure(df["TempInt"].mean)
print("\nPandas, mean() x%d" % REPEAT)
print("  float64:", round(t_float, 4), "с")
print("  int    :", round(t_int, 4), "с")

# --- Numpy ---
arr_float = df["AvgTemperature"].to_numpy(dtype=np.float64)
arr_int = df["TempInt"].to_numpy(dtype=np.int16)

t_float = measure(arr_float.mean)
t_int = measure(arr_int.mean)
print("\nNumpy, mean() x%d" % REPEAT)
print("  float64:", round(t_float, 4), "с")
print("  int16  :", round(t_int, 4), "с")

print("\nВывод: int16 не всегда быстрее. Для mean() numpy всё равно переводит")
print("целые числа в float64 при суммировании, поэтому выигрыша может не быть.")
