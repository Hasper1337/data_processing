"""ЛР №5: смещение и разброс, решающие деревья "своими руками", случайный лес.

Пункты 1-15 из блокнота homework-practice-03-ensembles (необязательные пункты со звёздочкой пропущены).
Датасеты UCI читаются из локальных файлов рядом со скриптом, а если их нет - скачиваются по ссылке.
Если датасет недоступен, соответствующий пункт пропускается (остальные выполняются).
Графики сохраняются в PNG.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import (BaggingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

np.random.seed(0)
pd.set_option("display.width", 200)          # чтобы таблицы не обрезались
pd.set_option("display.max_columns", None)
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


def load_table(filename, url, **kwargs):
    """Читает таблицу из локального файла или по ссылке; None, если не получилось."""
    for source in (filename, url):
        try:
            if source == filename and not os.path.exists(filename):
                continue
            return pd.read_csv(source, **kwargs)
        except Exception as error:
            print("   не удалось прочитать %s: %s" % (source, str(error)[:80]))
    return None


# =====================================================================
# I. Разложение ошибки на смещение и разброс
# =====================================================================
print("=== I. Смещение и разброс ===")


def f(x):
    return np.sin(x)           # истинная зависимость


sample_size = 100
samples_num = 20
linspace = np.linspace(0, 7, 1000)


def make_sample():
    x = np.random.exponential(size=sample_size)
    y = f(x) + np.random.uniform(-1, 1, size=sample_size)
    return x[:, np.newaxis], y


# Задача 1, п. 1: одно дерево
X, y = make_sample()
tree = DecisionTreeRegressor().fit(X, y)
plt.figure(figsize=(8, 4))
plt.plot(linspace, f(linspace), "r", label="f(x)")
plt.plot(linspace, tree.predict(linspace[:, np.newaxis]), "k", label="дерево")
plt.legend()
plt.savefig("task1_one_tree.png", dpi=110)
plt.close()

# Пункты 2-3: много деревьев с разной глубиной
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, depth in zip(axes, [None, 2, 4]):
    predictions = []
    for _ in range(samples_num):
        X, y = make_sample()
        t = DecisionTreeRegressor(max_depth=depth).fit(X, y)
        p = t.predict(linspace[:, np.newaxis])
        predictions.append(p)
        ax.plot(linspace, p, color="gray", alpha=0.5)
    ax.plot(linspace, f(linspace), color="red", label="f(x)")
    ax.plot(linspace, np.mean(predictions, axis=0), color="black", label="среднее по деревьям")
    ax.set_title("max_depth = %s" % depth)
    ax.legend()
plt.savefig("task1_many_trees.png", dpi=110)
plt.close()
print("1-3. графики сохранены: task1_one_tree.png, task1_many_trees.png")
print("4. Вывод: у глубокого дерева (без ограничений) среднее предсказание почти совпадает с f(x),")
print("   значит смещение близко к нулю (но разброс большой). При глубине 2 смещение велико,")
print("   при глубине 4 оно заметно меньше. Смещение мало у глубоких деревьев.")


def compute_bias_variance(regressor, dependence_fun, x_generator=np.random.uniform,
                          noise_generator=np.random.uniform, sample_size=100, samples_num=100,
                          objects_num=50):
    """Задача 2, п. 4: приближённая оценка смещения и разброса регрессора."""
    x = x_generator(size=objects_num)                    # объекты для оценки интеграла по x
    # E[y|x] = f(x) + E[шум], мат. ожидание шума оцениваем отдельной выборкой
    y_mean = dependence_fun(x) + noise_generator(size=objects_num).mean()

    def predictions():
        """Предсказания алгоритмов, обученных на samples_num разных выборках."""
        result = []
        for _ in range(samples_num):
            xs = x_generator(size=sample_size)
            ys = dependence_fun(xs) + noise_generator(size=sample_size)
            regressor.fit(xs[:, np.newaxis], ys)
            result.append(regressor.predict(x[:, np.newaxis]))
        return np.array(result)                         # форма (samples_num, objects_num)

    mean_prediction = predictions().mean(axis=0)         # оценка E_X[mu(X)](x)
    bias = np.mean((mean_prediction - y_mean) ** 2)
    # разброс: выборки генерируем заново
    variance = np.mean((predictions() - mean_prediction) ** 2)
    return bias, variance


# Пункт 5: таблицы
regressors = [LinearRegression(), DecisionTreeRegressor(),
              RandomForestRegressor(n_estimators=30), GradientBoostingRegressor()]
generators = [lambda size: np.random.normal(0, 0.3, size=size),
              lambda size: np.random.exponential(0.3, size=size),
              lambda size: np.random.uniform(-1, 1, size=size),
              np.random.uniform]
columns = ["normal(0, 0.3)", "exponential(0.3)", "uniform(-1, 1)", "uniform(0, 1)"]
rows = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting"]
bias = pd.DataFrame(columns=columns, index=rows, dtype=float)
variance = pd.DataFrame(columns=columns, index=rows, dtype=float)
for reg, row in zip(regressors, rows):
    for gen, col in zip(generators, columns):
        b, v = compute_bias_variance(reg, lambda x: 5 * np.sin(10 * x), gen, gen)
        bias.loc[row, col] = b
        variance.loc[row, col] = v
print("5. Смещение:\n", bias.round(3))
print("   Разброс:\n", variance.round(3))
print("6. Алгоритмы по возрастанию среднего разброса:",
      list(variance.mean(axis=1).sort_values().index))
print("   Смещение и разброс зависят не только от алгоритма, но и от распределения данных")
print("   (например, у экспоненциального распределения мало объектов в области больших x,")
print("   поэтому дерево плохо восстанавливает зависимость там).")

# =====================================================================
# II. Решающее дерево своими руками
# =====================================================================
print("\n=== II. Решающие деревья ===")


def find_best_split(feature_vector, target_vector):
    """Задача 3, п. 7: лучший порог по критерию Джини (без циклов).

    Возвращает: thresholds, ginis, оптимальный порог, оптимальное значение критерия.
    Q(R) = -|Rl|/|R| * H(Rl) - |Rr|/|R| * H(Rr),  H(R) = 1 - p1^2 - p0^2.
    """
    order = np.argsort(feature_vector, kind="stable")
    f_sorted = feature_vector[order]
    t_sorted = target_vector[order]

    change = np.where(f_sorted[1:] != f_sorted[:-1])[0]   # позиции, где значение меняется
    if len(change) == 0:                                  # признак константный
        return np.array([]), np.array([]), None, None

    thresholds = (f_sorted[change] + f_sorted[change + 1]) / 2
    n = len(t_sorted)
    n_left = change + 1
    n_right = n - n_left
    ones_total = t_sorted.sum()
    ones_left = np.cumsum(t_sorted)[change]
    ones_right = ones_total - ones_left

    p1_left = ones_left / n_left
    p1_right = ones_right / n_right
    h_left = 1 - p1_left ** 2 - (1 - p1_left) ** 2
    h_right = 1 - p1_right ** 2 - (1 - p1_right) ** 2
    ginis = -n_left / n * h_left - n_right / n * h_right

    best = np.argmax(ginis)
    return thresholds, ginis, thresholds[best], ginis[best]


# Проверка на простом примере
th, gi, bt, bg = find_best_split(np.array([1., 2., 3., 4.]), np.array([0, 0, 1, 1]))
print("7. проверка find_best_split:", bt, bg, "(ожидается порог 2.5 и Джини 0)")

# П. 8-9: User Knowledge
print("8. User Knowledge Modeling")
uk = None
for name in ["Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls", "user_knowledge.xls", "user_knowledge.csv"]:
    if os.path.exists(name):
        try:
            uk = pd.read_excel(name, sheet_name="Training_Data") if name.endswith("xls") else pd.read_csv(name)
        except Exception as error:
            print("   ошибка чтения:", str(error)[:100], "(для .xls нужен: pip install xlrd)")
        break
if uk is None:
    print("   файл с данными не найден - пункты 8-9 пропущены. Скачайте User Knowledge Modeling с UCI")
    print("   (файл .xls, лист Training_Data) и положите рядом со скриптом.")
else:
    uk = uk.iloc[:, :6]
    target_text = uk.iloc[:, 5].astype(str).str.lower().str.replace(" ", "_")
    y_uk = (~target_text.isin(["very_low", "low"])).astype(int).values   # 0: very_low/Low, 1: Middle/High
    X_uk = uk.iloc[:, :5].values.astype(float)
    plt.figure(figsize=(8, 5))
    best_feature, best_value = None, -np.inf
    for j in range(5):
        th, gi, bt, bg = find_best_split(X_uk[:, j], y_uk)
        plt.plot(th, gi, label=uk.columns[j])
        if bg is not None and bg > best_value:
            best_feature, best_value = uk.columns[j], bg
    plt.xlabel("порог")
    plt.ylabel("критерий Джини")
    plt.legend()
    plt.savefig("task8_gini_curves.png", dpi=110)
    plt.close()
    fig, axes = plt.subplots(1, 5, figsize=(20, 3))
    for j, ax in enumerate(axes):
        ax.scatter(X_uk[:, j], y_uk, s=8, alpha=0.5)
        ax.set_title(uk.columns[j])
    plt.savefig("task8_scatter.png", dpi=110)
    plt.close()
    print("   графики: task8_gini_curves.png, task8_scatter.png")
    print("9. Лучший признак для первого разбиения (макс. Джини):", best_feature, round(best_value, 3))
    print("   У хорошего признака кривая имеет резкий пик (Джини близок к 0), у плохого - почти")
    print("   плоская и далека от нуля.")


class DecisionTree(BaseEstimator):
    """Задача 3, п. 10: дерево для бинарной классификации.

    types[i] = 0 - признак вещественный, 1 - категориальный.
    Узел-список: [признак, порог (или множество категорий для левого поддерева), левое, правое];
    лист: [класс].
    """

    def __init__(self, types):
        self.types = types
        self._tree = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._tree = []

        def fit_node(sub_x, sub_y, node):
            if len(np.unique(sub_y)) == 1:              # все объекты одного класса
                node.append(sub_y[0])
                return

            best = None                                 # (Джини, признак, порог, категории слева)
            for j in range(sub_x.shape[1]):
                column = sub_x[:, j]
                if self.types[j] == 0:
                    feature = column.astype(float)
                    cats_order = None
                else:
                    # категориальный признак: упорядочиваем категории по доле класса 1
                    cats, inverse = np.unique(column, return_inverse=True)
                    ratio = np.bincount(inverse, weights=sub_y) / np.bincount(inverse)
                    rank = np.argsort(np.argsort(ratio))
                    feature = rank[inverse].astype(float)
                    cats_order = cats[np.argsort(ratio)]       # категории в порядке рангов
                _, _, threshold, gini = find_best_split(feature, sub_y)
                if threshold is None:
                    continue
                if best is None or gini > best[0]:
                    left_cats = None if cats_order is None else set(cats_order[:int(np.ceil(threshold))])
                    best = (gini, j, threshold, left_cats)

            if best is None:                            # разделить нельзя -> лист с большинством
                node.append(np.bincount(sub_y.astype(int)).argmax())
                return

            _, j, threshold, left_cats = best
            if left_cats is None:
                mask = sub_x[:, j].astype(float) < threshold
                node.extend([j, threshold])
            else:
                mask = np.isin(sub_x[:, j], list(left_cats))
                node.extend([j, left_cats])
            left, right = [], []
            node.extend([left, right])
            fit_node(sub_x[mask], sub_y[mask], left)
            fit_node(sub_x[~mask], sub_y[~mask], right)

        fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        def predict_node(x, node):
            if len(node) == 1:
                return node[0]
            j, rule, left, right = node
            if self.types[j] == 0:
                go_left = float(x[j]) < rule
            else:
                go_left = x[j] in rule
            return predict_node(x, left if go_left else right)

        return np.array([predict_node(x, self._tree) for x in np.asarray(X)])


# Проверка на синтетике: категориальный признак, который нельзя разделить порогом
Xs = np.array([[0], [1], [2], [3], [4], [5]] * 10)
ys = np.array([0, 1, 0, 1, 0, 1] * 10)
print("10. проверка дерева (категориальный признак):",
      accuracy_score(ys, DecisionTree([1]).fit(Xs, ys).predict(Xs)),
      "| как вещественный:", accuracy_score(ys, DecisionTree([0]).fit(Xs, ys).predict(Xs)))

# П. 11-13: наборы данных UCI
print("11-13. Сравнение на наборах данных UCI")


def encode_all(frame):
    return frame.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))


datasets = {}
mush = load_table("agaricus-lepiota.data", UCI + "mushroom/agaricus-lepiota.data", header=None)
if mush is not None:
    mush = encode_all(mush)
    datasets["mushrooms"] = (mush.iloc[:, 1:].values, mush.iloc[:, 0].values)
ttt = load_table("tic-tac-toe.data", UCI + "tic-tac-toe/tic-tac-toe.data", header=None)
if ttt is not None:
    ttt = encode_all(ttt)
    datasets["tic-tac-toe"] = (ttt.iloc[:, :-1].values, ttt.iloc[:, -1].values)
cars = load_table("car.data", UCI + "car/car.data", header=None)
if cars is not None:
    y_cars = cars.iloc[:, -1].isin(["good", "vgood"]).astype(int).values
    datasets["cars"] = (encode_all(cars.iloc[:, :-1]).values, y_cars)
nursery = load_table("nursery.data", UCI + "nursery/nursery.data", header=None)
if nursery is not None:
    y_nur = nursery.iloc[:, -1].isin(["very_recom", "priority", "spec_prior"]).astype(int).values
    datasets["nursery"] = (encode_all(nursery.iloc[:, :-1]).values, y_nur)

if "mushrooms" in datasets:
    X_m, y_m = datasets["mushrooms"]
    Xa, Xb, ya, yb = train_test_split(X_m, y_m, test_size=0.5, random_state=0)
    clf = DecisionTree([1] * X_m.shape[1]).fit(Xa, ya)
    print("11. mushrooms: accuracy на контроле =", accuracy_score(yb, clf.predict(Xb)))

if datasets:
    scorer = make_scorer(accuracy_score)
    table = pd.DataFrame(index=list(datasets), columns=[
        "DT вещественные", "DT категориальные", "DT вещ. + one-hot", "sklearn DecisionTreeClassifier"], dtype=float)
    for name, (X_d, y_d) in datasets.items():
        n_features = X_d.shape[1]
        X_onehot = OneHotEncoder(sparse_output=False).fit_transform(X_d)
        table.loc[name, "DT вещественные"] = cross_val_score(
            DecisionTree([0] * n_features), X_d, y_d, cv=10, scoring=scorer).mean()
        table.loc[name, "DT категориальные"] = cross_val_score(
            DecisionTree([1] * n_features), X_d, y_d, cv=10, scoring=scorer).mean()
        table.loc[name, "DT вещ. + one-hot"] = cross_val_score(
            DecisionTree([0] * X_onehot.shape[1]), X_onehot, y_d, cv=10, scoring=scorer).mean()
        table.loc[name, "sklearn DecisionTreeClassifier"] = cross_val_score(
            DecisionTreeClassifier(), X_d, y_d, cv=10, scoring=scorer).mean()
    print("12. accuracy (cv=10):\n", table.round(4))
    print("13. Результаты зависят от набора: на признаках с естественным порядком вещественная")
    print("    трактовка работает хорошо, на чисто категориальных лучше категориальная или one-hot.")
    print("    Случайность есть в разбиении на фолды и в выборе при равных Джини; можно зафиксировать")
    print("    random_state и перемешать данные (KFold(shuffle=True)).")
else:
    print("   наборы данных UCI недоступны - пункты 11-13 пропущены (положите файлы рядом со скриптом).")

# =====================================================================
# III. Композиции деревьев
# =====================================================================
print("\n=== III. Композиции деревьев ===")
wine = load_table("winequality-red.csv", UCI + "wine-quality/winequality-red.csv", sep=";")
if wine is None:
    print("   winequality-red.csv недоступен - пункты 14-15 пропущены.")
else:
    X_w, y_w = wine.iloc[:, :-1].values, wine.iloc[:, -1].values
    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "Bagging (100 деревьев)": BaggingClassifier(n_estimators=100, n_jobs=-1),
        "Bagging + случайные признаки (0.5)": BaggingClassifier(n_estimators=100, max_features=0.5, n_jobs=-1),
        "RandomForest (100 деревьев)": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    }
    print("14. accuracy (cv=3):")
    for name, model in models.items():
        print("   %-36s %.4f" % (name, cross_val_score(model, X_w, y_w, cv=3, scoring="accuracy").mean()))

    # Задача 5: число деревьев
    X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.3, shuffle=True, random_state=0)
    forest = RandomForestClassifier(n_estimators=100, warm_start=True, n_jobs=-1, random_state=0)
    counts = list(range(100, 5001, 100))
    scores = []
    for n_trees in counts:
        forest.set_params(n_estimators=n_trees)
        forest.fit(X_tr, y_tr)                  # достраивает только недостающие деревья
        scores.append(accuracy_score(y_te, forest.predict(X_te)))
    plt.figure(figsize=(8, 4))
    plt.plot(counts, scores)
    plt.xlabel("число деревьев")
    plt.ylabel("accuracy на контроле")
    plt.grid()
    plt.savefig("task15_forest_size.png", dpi=110)
    plt.close()
    print("15. график task15_forest_size.png: accuracy сначала растёт, затем выходит на плато и не падает.")
    print("    Случайный лес не переобучается с ростом числа деревьев (ошибка стабилизируется).")
