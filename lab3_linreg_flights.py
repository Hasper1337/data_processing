"""ЛР №3: NumPy, Pandas, линейная регрессия и градиентный спуск (данные 2008.csv, авиарейсы).

Запуск:  python lab3_linreg_flights.py 2008.csv [airports.csv]
Файл airports.csv (iata, city) нужен только для ответа "в каком городе находится аэропорт" (п. 9).
Графики сохраняются в PNG рядом со скриптом, ответы печатаются в консоль.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

DATA = sys.argv[1] if len(sys.argv) > 1 else "2008.csv"
AIRPORTS = sys.argv[2] if len(sys.argv) > 2 else "airports.csv"
NROWS = 100000          # сколько строк читать (весь файл очень большой и медленный)
GD_TRAIN_SIZE = 5000    # сколько объектов использовать в градиентном спуске (для скорости)

# =====================================================================
# NumPy (без циклов)
# =====================================================================


def max_element(arr):
    """П.1: максимум среди элементов, перед которыми стоит нулевой."""
    after_zero = arr[1:][arr[:-1] == 0]
    return after_zero.max() if after_zero.size > 0 else None


def nearest_value(X, v):
    """П.2: ближайший к v элемент матрицы."""
    return X.flat[np.abs(X - v).argmin()]


def scale(X):
    """П.3: вычесть среднее и поделить на стандартное отклонение по столбцам."""
    std = X.std(axis=0)
    std[std == 0] = 1          # чтобы не делить на ноль
    return (X - X.mean(axis=0)) / std


def get_stats(X):
    """П.4: характеристики матрицы."""
    return {
        "определитель": np.linalg.det(X),
        "след": np.trace(X),
        "минимум": X.min(),
        "максимум": X.max(),
        "норма Фробениуса": np.linalg.norm(X, "fro"),
        "собственные числа": np.linalg.eigvals(X),
        "обратная матрица": np.linalg.inv(X),
    }


print("=== NumPy ===")
print("1.", max_element(np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])), "(ожидается 5)")
print("2.", nearest_value(np.arange(0, 10).reshape((2, 5)), 3.6), "(ожидается 4)")
Xr = np.random.randint(0, 10, size=(5, 3)).astype(float)
print("3. средние после scale:", scale(Xr).mean(axis=0).round(6), "std:", scale(Xr).std(axis=0).round(6))
stats = get_stats(np.random.normal(10, 1, size=(4, 4)))
for k, v in stats.items():
    if k not in ("обратная матрица", "собственные числа"):
        print("4.", k, "=", v)

# П.5: 100 экспериментов сразу (массив 100 x 10 x 10), без цикла
A = np.random.standard_normal((100, 10, 10))
B = np.random.standard_normal((100, 10, 10))
maxima = (A @ B).max(axis=(1, 2))
print("5. среднее максимумов:", maxima.mean(), "| 95%-квантиль:", np.quantile(maxima, 0.95))

# =====================================================================
# Pandas
# =====================================================================
print("\n=== Pandas ===")
df = pd.read_csv(DATA, nrows=NROWS)
print("Размер:", df.shape)

print("6. самая частая причина отмены:", df["CancellationCode"].value_counts().idxmax(),
      "(A - перевозчик, B - погода, C - NAS, D - безопасность)")
print("7. расстояние: среднее", df["Distance"].mean(), "мин", df["Distance"].min(), "макс", df["Distance"].max())

short = df[df["Distance"] == df["Distance"].min()]
print("8. рейсы с минимальным расстоянием:")
print(short[["Month", "DayofMonth", "UniqueCarrier", "FlightNum", "Origin", "Dest", "Distance"]].head())
same = df[df["FlightNum"].isin(short["FlightNum"]) & df["UniqueCarrier"].isin(short["UniqueCarrier"])]
print("   те же рейсы в другие дни (Distance):")
print(same.groupby(["UniqueCarrier", "FlightNum"])["Distance"].agg(["min", "max", "count"]).head())

top_airport = df["Origin"].value_counts().idxmax()
print("9. больше всего вылетов из:", top_airport, end="")
try:
    ap = pd.read_csv(AIRPORTS)
    print(", город:", ap.loc[ap["iata"] == top_airport, "city"].values)
except FileNotFoundError:
    print(" (для города нужен airports.csv)")

air = df.groupby("Origin")["AirTime"].mean()
print("10. наибольшее среднее AirTime:", air.idxmax(), round(air.max(), 1))

big = df.groupby("Origin").filter(lambda g: len(g) >= 1000)
if len(big) > 0:
    share = (big["DepDelay"] > 0).groupby(big["Origin"]).mean()
    print("11. наибольшая доля задержек:", share.idxmax(), round(share.max(), 3))
else:
    print("11. нет аэропортов с 1000+ рейсов (увеличьте NROWS)")

# =====================================================================
# Подготовка данных
# =====================================================================
print("\n=== Подготовка данных ===")
# 12. Пропуски
print("12. есть пропуски:", df.isna().any().any())
print("    всего пропущенных элементов:", int(df.isna().sum().sum()))
print("    объектов с пропуском:", int(df.isna().any(axis=1).sum()))
print("    признаков с пропуском:", int(df.isna().any(axis=0).sum()))
print("    пропусков в DepDelay:", int(df["DepDelay"].isna().sum()))
missing = df[df["DepDelay"].isna()]
print("    у объектов без DepDelay Cancelled == 1 в доле:", (missing["Cancelled"] == 1).mean())
# Пропуск DepDelay бывает только у отменённых рейсов (они не вылетели), поэтому
# признаки Cancelled и CancellationCode после удаления таких объектов бесполезны.

df = df.dropna(subset=["DepDelay"])
df = df[df["DepDelay"] != 0]
y = df["DepDelay"].astype(float)
X = df.drop(columns=["DepDelay", "Cancelled", "CancellationCode"])

# 13. hhmm -> час и минута
for col in ["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]:
    X[col + "_Hour"] = X[col] // 100          # NaN остаётся NaN
    X[col + "_Minute"] = X[col] % 100
    X = X.drop(columns=col)

# 14. Исключаем признаки, которые "выдают" ответ:
#  - DepTime (фактическое время вылета): DepDelay = DepTime - CRSDepTime;
#  - ArrTime, ArrDelay, ActualElapsedTime, AirTime, TaxiIn, Diverted и причины задержек
#    становятся известны только после полёта и напрямую связаны с задержкой.
leak = ["DepTime_Hour", "DepTime_Minute", "ArrTime_Hour", "ArrTime_Minute", "ArrDelay",
        "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
        "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "TailNum", "Year"]
X = X.drop(columns=[c for c in leak if c in X.columns])
print("14. оставшиеся признаки:", list(X.columns))

# 15. Масштабирование и one-hot кодирование
for col in ["CRSDepTime_Hour", "TaxiOut", "FlightNum"]:
    plt.figure()
    X[col].hist(bins=20)
    plt.title(col)
    plt.savefig("hist_%s.png" % col)
    plt.close()
print("15. гистограммы сохранены (hist_*.png): у признаков разный масштаб, у FlightNum")
print("    значения до 7000, у остальных - десятки; без масштабирования веса несравнимы.")


def transform_data(data):
    data = data.copy()
    cat_cols = list(data.select_dtypes(exclude="number").columns)
    num_cols = [c for c in data.columns if c not in cat_cols]
    data[num_cols] = data[num_cols].fillna(0)
    data[cat_cols] = data[cat_cols].fillna("nan").astype(str)
    data[num_cols] = StandardScaler().fit_transform(data[num_cols])
    return pd.get_dummies(data, columns=cat_cols, dtype=float)


X = transform_data(X)
print("    признаков после преобразования:", X.shape[1])

# 16. Разбиение 70/30
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=0)
print("16. обучение:", X_train.shape, "контроль:", X_test.shape)

# =====================================================================
# Scikit-learn
# =====================================================================
print("\n=== Scikit-learn ===")


def report(model, name, Xa, ya, Xb, yb):
    pa, pb = model.predict(Xa), model.predict(Xb)
    print("%-10s train MSE=%.1f R2=%.3f | test MSE=%.1f R2=%.3f" % (
        name, mean_squared_error(ya, pa), r2_score(ya, pa), mean_squared_error(yb, pb), r2_score(yb, pb)))


Xs, ys = X_train[:1000], y_train[:1000]
lin = LinearRegression().fit(Xs, ys)
print("17.")
report(lin, "Linear", Xs, ys, X_test, y_test)
print("    max |w| =", np.abs(lin.coef_).max(), "- огромные веса и переобучение (признаков больше, чем объектов)")

alpha_grid = np.logspace(-3, 3, 10)
ridge = RidgeCV(alphas=alpha_grid, cv=5).fit(Xs, ys)
lasso = LassoCV(alphas=alpha_grid, cv=KFold(5), max_iter=5000).fit(Xs, ys)
print("18.")
report(ridge, "Ridge", Xs, ys, X_test, y_test)
report(lasso, "Lasso", Xs, ys, X_test, y_test)
print("    лучшие alpha: Ridge", ridge.alpha_, "Lasso", lasso.alpha_, "| max |w| Ridge", np.abs(ridge.coef_).max())

# =====================================================================
# Градиентный спуск
# =====================================================================
print("\n=== Градиентный спуск ===")
# 19. Единичный признак
Xtr = np.hstack([X_train, np.ones((len(X_train), 1))])[:GD_TRAIN_SIZE]
ytr = y_train[:GD_TRAIN_SIZE]
Xte = np.hstack([X_test, np.ones((len(X_test), 1))])
yte = y_test


# 20. Функции
def make_pred(X, w):
    return X @ w


def get_func(w, X, y):
    """MSPE = mean(((y - Xw) / y)^2)."""
    return np.mean(((y - make_pred(X, w)) / y) ** 2)


def get_grad(w, X, y):
    """Градиент MSPE: 2/l * X^T (Xw - y) / y^2 (деление на y^2 поэлементное)."""
    return 2 / len(y) * X.T @ ((make_pred(X, w) - y) / y ** 2)


def get_reg_func(w, X, y):
    return get_func(w, X, y) + np.sum(w ** 2)


def get_reg_grad(w, X, y):
    return get_grad(w, X, y) + 2 * w


# 21. Градиентный спуск
def grad_descent(X, y, w0, step_size, max_iter, eps, is_reg):
    func = get_reg_func if is_reg else get_func
    grad = get_reg_grad if is_reg else get_grad
    w = w0.copy()
    history = [func(w, X, y)]
    for _ in range(max_iter):
        w_new = w - step_size * grad(w, X, y)
        history.append(func(w_new, X, y))
        if not np.isfinite(history[-1]):       # метод расходится
            break
        done = np.linalg.norm(w_new - w) < eps
        w = w_new
        if done:
            break
    return w, history


# 23. Стохастический градиентный спуск (один случайный объект за шаг)
def sgd(X, y, w0, step_size, max_iter, eps, is_reg, seed=0):
    rng = np.random.default_rng(seed)
    func = get_reg_func if is_reg else get_func
    grad = get_reg_grad if is_reg else get_grad
    w = w0.copy()
    history = [func(w, X, y)]
    for k in range(1, max_iter + 1):
        i = rng.integers(len(y))
        w_new = w - step_size * grad(w, X[i:i + 1], y[i:i + 1])
        if k % 100 == 0:                       # функционал на всей выборке считаем редко
            history.append(func(w_new, X, y))
            if not np.isfinite(history[-1]):
                break
        done = np.linalg.norm(w_new - w) < eps
        w = w_new
        if done:
            break
    return w, history


def plot_history(histories, title, filename):
    plt.figure(figsize=(8, 5))
    for label, h in histories.items():
        plt.plot(h, label=label)
    plt.yscale("log")
    plt.xlabel("итерация")
    plt.ylabel("значение функционала")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename, dpi=110)
    plt.close()


d = Xtr.shape[1]
rng = np.random.default_rng(0)
inits = {"нули": np.zeros(d), "случайные": rng.normal(0, 0.1, d)}

# Эксперимент с шагом и начальными весами
print("21. MSPE на обучении после 100 итераций:")
hists = {}
with np.errstate(all="ignore"):
    for step in [0.001, 1, 10]:
        for name, w0 in inits.items():
            w, h = grad_descent(Xtr, ytr, w0, step, 100, 1e-6, False)
            hists["шаг %g, %s" % (step, name)] = h
            print("    шаг %-6g w0=%-10s -> %.4g (итераций %d)" % (step, name, h[-1], len(h) - 1))
plot_history(hists, "Градиентный спуск, MSPE", "gd_steps.png")

# 22. С регуляризацией и без
STEP = 0.001
with np.errstate(all="ignore"):
    w_plain, h_plain = grad_descent(Xtr, ytr, np.zeros(d), STEP, 300, 1e-8, False)
    w_reg, h_reg = grad_descent(Xtr, ytr, np.zeros(d), STEP, 300, 1e-8, True)
plot_history({"MSPE": h_plain, "MSPE + L2": h_reg}, "Сходимость с регуляризацией и без", "gd_reg.png")
print("22. итоговый функционал без/с регуляризацией:", h_plain[-1], h_reg[-1])
print("    регуляризация штрафует большие веса, поэтому веса остаются меньше, а функционал выше.")

# 23-24. SGD
with np.errstate(all="ignore"):
    w_sgd, h_sgd = sgd(Xtr, ytr, np.zeros(d), 0.001, 20000, 1e-12, False)
    w_sgd_reg, h_sgd_reg = sgd(Xtr, ytr, np.zeros(d), 0.001, 20000, 1e-12, True)
plot_history({"SGD": h_sgd, "SGD + L2": h_sgd_reg}, "SGD (точки через 100 шагов)", "sgd.png")
pred = make_pred(Xte, w_sgd)
print("23. SGD на контроле: MSPE=%.4g MSE=%.1f R2=%.3f" % (
    get_func(w_sgd, Xte, yte), mean_squared_error(yte, pred), r2_score(yte, pred)))
print("24. итог SGD без/с регуляризацией:", h_sgd[-1], h_sgd_reg[-1])

# 25. Обычная линейная регрессия (MSE) и её MSPE на контроле
lin_full = LinearRegression().fit(X_train, y_train)
w_lin = np.append(lin_full.coef_, lin_full.intercept_)
print("25. MSPE на контроле: МНК (MSE) =", get_func(w_lin, Xte, yte),
      "| градиентный спуск по MSPE =", get_func(w_plain, Xte, yte))
print("    МНК минимизирует MSE, а не MSPE, поэтому по MSPE он обычно хуже решения, найденного прямой оптимизацией MSPE.")
