import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
# def max_element(arr):
#   zero = np.where(arr == 0)[0]
#   if len(zero) == 0:
#     return None

#   after_zero = zero + 1
#   valid = after_zero[after_zero < len(arr)]
#   values = arr[valid]
#   return np.max(values)

# print(max_element(x))
# print("#########################################")
# X = np.arange(0,10).reshape((2, 5))
# v = 3.6
# def nearest_val(X, v):
#   diff = np.abs(X - v)
#   min_index = np.argmin(diff)
#   return X.ravel()[min_index]

# print(nearest_val(X, v))
# print("#########################################")
# x = np.random.randint(5, size=(4, 4))
# def scale(X):
#   stds = np.std(X, axis=0)
#   stds[stds == 0] = 1
#   X_scl = (X - np.mean(X, axis=0)) / stds
#   return X_scl

# print(scale(x))
# print("#########################################")
# x = np.random.randint(5, size=(4, 4))
# def get_stats(X):
#   print(f'Матрица:\n {X}')
#   print(f'Определитель матрицы: {np.linalg.det(X)}')
#   print(f'След матрицы: {np.trace(X)}')
#   print(f'Наименьший и наибольший элементы матрицы: {np.min(X)}, {np.max(X)}')
#   print(f'Норма Фробениуса: {np.linalg.norm(X, ord='fro')}')
#   print(f'Собственные числа матрицы:\n {np.linalg.eig(X)}')
#   print(f'Обратная матрица:\n {np.linalg.inv(X)}')

# get_stats(x)
# print("#########################################")
# x10 = 0
# for exp_num in range(100):
#   x1 = np.random.randint(10, size=(10, 10))
#   x2 = np.random.randint(10, size=(10, 10))
#   x3 = x1 * x2
#   max_index = np.argmax(x3)
#   x10 += x3.ravel()[max_index]
#   print(x3.ravel()[max_index])
# print(x10/100, np.quantile(x3, 0.95))

airline = pd.read_csv("2008.csv")

# #6. Какая из причин отмены рейса (CancellationCode) была самой частой? (расшифровки кодов можно найти в описании данных)
# print(airline['CancellationCode'].value_counts()[:1])
# #7. (0.3 балла) Найдите среднее, минимальное и максимальное расстояние, пройденное самолетом.
# print(f"Найдите среднее, минимальное и максимальное расстояние:\nmean: {airline['Distance'].mean():.2f}, min: {airline['Distance'].min()}, max: {airline['Distance'].max()}")
# #8. (0.3 балла) Не выглядит ли подозрительным минимальное пройденное расстояние? В какие дни и на каких рейсах оно было? Какое расстояние было пройдено этими же рейсами в другие дни?
# fil_air = airline[airline['Distance'] == 31]
# col = ['Year', 'Month', 'DayofMonth', 'Distance', 'FlightNum']
# print(fil_air[col])
# FlightNum64_Airline = airline[airline['FlightNum'] == 64]
# FlightNum65_Airline = airline[airline['FlightNum'] == 65]
# print(FlightNum64_Airline[col][:5])
# print(FlightNum65_Airline[col][:5])
# #9. (0.3 балла) Из какого аэропорта было произведено больше всего вылетов? В каком городе он находится?
# print(f'\n{airline['Origin'].value_counts()[:1]}') # Атланты, Джорджия, США
# #10. (0.3 балла) Найдите для каждого аэропорта среднее время полета (AirTime) по всем вылетевшим из него рейсам. Какой аэропорт имеет наибольшее значение этого показателя?
# print(airline.groupby('Origin')['AirTime'].mean())
# print(f"Аэропорт с наибольшим средним временем полета: {airline.groupby('Origin')['AirTime'].mean().idxmax()} {airline.groupby('Origin')['AirTime'].mean().max()}")
# #11. (0.5 балла) Найдите аэропорт, у которого наибольшая доля задержанных (DepDelay > 0) рейсов. Исключите при этом из рассмотрения аэропорты, из которых было отправлено меньше 1000 рейсов (используйте функцию filter после groupby).
# dep_airline = airline[airline['DepDelay'] > 0]
# dep1_airline = dep_airline.groupby('Origin').filter(lambda x: len(x) >= 1000)
# print(dep1_airline['Origin'].value_counts()[:1])

# Имеются ли в данных пропущенные значения?
print(airline.isnull().values.any())

# Сколько всего пропущенных элементов в таблице "объект-признак"?
print(airline.isnull().sum().sum())

# Сколько объектов имеют хотя бы один пропуск?
print(airline.isnull().any(axis=1).sum())

# Сколько признаков имеют хотя бы одно пропущенное значение?
print(airline.isnull().any(axis=0).sum())

# Имеются ли пропущенные значения в целевой переменной?
target_per = 'DepDelay'
print(airline[target_per].isnull().sum())

# Проверка связи с другими признаками
missing_target_rows = airline[airline[target_per].isnull()]
if len(missing_target_rows) > 0:
    cancelled_in_missing = missing_target_rows['Cancelled'].sum()
    print(f"Из них отмененных рейсов: {cancelled_in_missing}")

# Исключаем объекты с пропущенным значением целевой переменной
airline_clean = airline.dropna(subset=[target_per]).copy()

# Исключаем объекты со значением целевой переменной, равным 0
airline_clean = airline_clean[airline_clean[target_per] != 0].copy()

leakage_features = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'TailNum', 'Year', 'ArrDelay']
airline_clean = airline_clean.drop(columns=leakage_features)

# 4. Выделение целевой переменной
y = airline_clean[target_per].values
X = airline_clean.drop(columns=[target_per])

print(f"\nРазмер матрицы признаков до очистки {airline.shape}, после {X.shape}")
print(f"Размер вектора целевой переменной: {y.shape}")

# 13. (0.5 балла) Обратите внимание, что признаки DepTime, CRSDepTime, ArrTime, CRSArrTime приведены в формате hhmm, в связи с чем будет не вполне корректно рассматривать их как вещественные.
time_features = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime']

for feature in time_features:
    hour_col = f"{feature}_Hour"
    minute_col = f"{feature}_Minute"

    not_na_mask = airline_clean[feature].notna()

    valid_values = airline_clean.loc[not_na_mask, feature].astype(int)

    airline_clean.loc[not_na_mask, hour_col] = valid_values // 100
    airline_clean.loc[not_na_mask, minute_col] = valid_values % 100

    airline_clean.drop(columns=[feature], inplace=True)

# Проверка результата
print(airline_clean[['DepTime_Hour', 'DepTime_Minute', 'CRSDepTime_Hour', 'CRSDepTime_Minute']].head(10))
# каждую гистограмму рисуем в отдельном окне, иначе они накладываются друг на друга
for col in ['DepTime_Hour', 'TaxiIn', 'FlightNum']:
    plt.figure()
    airline_clean[col].hist(bins=20)
    plt.title(col)
plt.show()


# 14. Исключаем признаки, которые слишком сильно связаны с целевой переменной
# Важно: в п. 13 столбцы со временем менялись в airline_clean, а X был выделен раньше,
# поэтому X нужно получить заново.
X = airline_clean.drop(columns=[target_per])

# Посмотрим на корреляцию вещественных признаков с DepDelay
with np.errstate(all='ignore'):  # у константных столбцов корреляция не определена (NaN)
    print(X.select_dtypes(include='number').corrwith(airline_clean[target_per]).sort_values(ascending=False))

# Обоснование:
# - DepTime_Hour, DepTime_Minute - фактическое время вылета. DepDelay = DepTime - CRSDepTime,
#   то есть ответ по этим признакам просто вычисляется.
# - ArrTime_*, ActualElapsedTime, AirTime, TaxiIn, TaxiOut, Diverted - фактические значения,
#   они становятся известны только после вылета/прилёта, а задержку мы хотим предсказывать заранее.
# - Cancelled, CancellationCode - после удаления пропусков DepDelay отменённых рейсов не осталось
#   (Cancelled всегда 0, CancellationCode всегда пустой), поэтому эти признаки бесполезны.
leakage_features_2 = ['DepTime_Hour', 'DepTime_Minute', 'ArrTime_Hour', 'ArrTime_Minute',
                      'ActualElapsedTime', 'AirTime', 'TaxiIn', 'TaxiOut', 'Diverted',
                      'Cancelled', 'CancellationCode']
X = X.drop(columns=leakage_features_2)
print(f"\nОставшиеся признаки: {list(X.columns)}")

# 15. Масштабирование вещественных признаков и one-hot кодирование категориальных
# По гистограммам видно, что признаки имеют совершенно разный масштаб (часы 0..24,
# TaxiIn - десятки минут, FlightNum - тысячи). Без масштабирования веса линейной модели
# нельзя сравнивать между собой, а градиентный спуск будет сходиться плохо.

# Во всём файле около 7 млн строк, после one-hot кодирования (сотни столбцов) это не поместится
# в память, поэтому берём случайную подвыборку.
SAMPLE_SIZE = 50000
if len(X) > SAMPLE_SIZE:
    X = X.sample(n=SAMPLE_SIZE, random_state=42)
y = airline_clean.loc[X.index, target_per].values


def transform_data(data):
    data = data.copy()
    num_cols = data.select_dtypes(include='number').columns
    cat_cols = data.select_dtypes(exclude='number').columns

    # 1) пропуски: нули в вещественных, строка 'nan' в категориальных
    data[num_cols] = data[num_cols].fillna(0)
    data[cat_cols] = data[cat_cols].fillna('nan')

    # 2) масштабирование вещественных признаков
    data[num_cols] = StandardScaler().fit_transform(data[num_cols])

    # 3) one-hot кодирование категориальных признаков
    return pd.get_dummies(data, columns=cat_cols, dtype=float)


X = transform_data(X)
print(f"Размер матрицы после transform_data: {X.shape}")

# 16. Разбиение на обучающую и контрольную выборки 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Обучение: {X_train.shape}, контроль: {X_test.shape}")

# 17. Линейная регрессия на первых 1000 объектах обучающей выборки
X_train_1000, y_train_1000 = X_train[:1000], y_train[:1000]
lin_reg = LinearRegression().fit(X_train_1000, y_train_1000)

pred_train = lin_reg.predict(X_train_1000)
pred_test = lin_reg.predict(X_test)
print("\nLinearRegression:")
print(f"train: MSE = {mean_squared_error(y_train_1000, pred_train):.2f}, R2 = {r2_score(y_train_1000, pred_train):.3f}")
print(f"test:  MSE = {mean_squared_error(y_test, pred_test):.2f}, R2 = {r2_score(y_test, pred_test):.3f}")
print(f"Максимальный по модулю вес: {np.abs(lin_reg.coef_).max():.3e}")
# Признаков (сотни после one-hot) сравнимо с числом объектов (1000), поэтому модель переобучается:
# на обучении ошибка маленькая, а на контроле большая; веса могут получиться огромными.

# 18. Ridge (L2) и Lasso (L1) с подбором alpha по кросс-валидации на 5 фолдах
alpha_grid = np.logspace(-3, 3, 10)

ridge_search = GridSearchCV(Ridge(), {'alpha': alpha_grid}, cv=5, scoring='neg_mean_squared_error')
ridge_search.fit(X_train_1000, y_train_1000)
ridge = ridge_search.best_estimator_

lasso_search = GridSearchCV(Lasso(max_iter=10000), {'alpha': alpha_grid}, cv=5, scoring='neg_mean_squared_error')
lasso_search.fit(X_train_1000, y_train_1000)
lasso = lasso_search.best_estimator_

for name, model in [('Ridge', ridge), ('Lasso', lasso)]:
    pred_test = model.predict(X_test)
    print(f"\n{name}: лучшее alpha = {model.alpha:.3g}")
    print(f"test: MSE = {mean_squared_error(y_test, pred_test):.2f}, R2 = {r2_score(y_test, pred_test):.3f}")
    print(f"Максимальный по модулю вес: {np.abs(model.coef_).max():.3f}")
print(f"Нулевых весов у Lasso: {np.sum(lasso.coef_ == 0)} из {len(lasso.coef_)}")
# Регуляризация ограничивает веса, поэтому модели меньше переобучаются.
# Lasso (L1) зануляет часть весов, то есть сам отбирает признаки.

# 19. Градиентный спуск для MSPE. Добавляем единичный признак (свободный член)
X_train_gd = np.hstack([X_train.values, np.ones((len(X_train), 1))])
X_test_gd = np.hstack([X_test.values, np.ones((len(X_test), 1))])

# 20. Функции: предсказание, MSPE и его градиент, а также версии с L2-регуляризацией
REG_COEF = 0.01


def make_pred(X, w):
    return X @ w


def get_func(w, X, y):
    # MSPE = 1/l * sum(((y - Xw) / y)^2)
    return np.mean(((y - make_pred(X, w)) / y) ** 2)


def get_grad(w, X, y):
    # производная MSPE по w: 2/l * X^T * ((Xw - y) / y^2)
    return 2 / len(y) * X.T @ ((make_pred(X, w) - y) / y ** 2)


def get_reg_func(w, X, y):
    return get_func(w, X, y) + REG_COEF * np.sum(w ** 2)


def get_reg_grad(w, X, y):
    return get_grad(w, X, y) + 2 * REG_COEF * w


# 21. Градиентный спуск
def grad_descent(X, y, w0, step_size, max_iter, eps, is_reg):
    if is_reg:
        func, grad = get_reg_func, get_reg_grad
    else:
        func, grad = get_func, get_grad

    w = w0.copy()
    history = [func(w, X, y)]
    for i in range(max_iter):
        w_new = w - step_size * grad(w, X, y)
        history.append(func(w_new, X, y))
        if not np.isfinite(history[-1]):        # шаг слишком большой - метод разошёлся
            break
        if np.linalg.norm(w_new - w) < eps:     # веса почти не меняются - останавливаемся
            w = w_new
            break
        w = w_new
    return w, history


n_features = X_train_gd.shape[1]
w_zeros = np.zeros(n_features)
w_random = np.random.default_rng(0).normal(0, 0.1, n_features)

plt.figure(figsize=(9, 5))
for step in [0.0001, 0.001, 0.01, 0.1]:
    for w0_name, w0 in [('нули', w_zeros), ('случайные', w_random)]:
        with np.errstate(all='ignore'):
            w, history = grad_descent(X_train_gd, y_train, w0, step, 300, 1e-6, False)
        print(f"шаг = {step}, начальные веса = {w0_name}: MSPE = {history[-1]:.4g}, итераций = {len(history) - 1}")
        plt.plot(history, label=f"шаг {step}, w0 = {w0_name}")
plt.yscale('log')
plt.xlabel('итерация')
plt.ylabel('MSPE на обучении')
plt.title('Градиентный спуск при разных шагах и начальных весах')
plt.legend()
plt.show()
# Слишком маленький шаг - медленная сходимость, слишком большой - метод расходится.
# От начального приближения результат почти не зависит (функционал выпуклый).

# 22. Сходимость с регуляризацией и без
STEP = 0.001
w_gd, history_gd = grad_descent(X_train_gd, y_train, w_zeros, STEP, 1000, 1e-6, False)
w_gd_reg, history_gd_reg = grad_descent(X_train_gd, y_train, w_zeros, STEP, 1000, 1e-6, True)

plt.figure(figsize=(9, 5))
plt.plot(history_gd, label='без регуляризации')
plt.plot(history_gd_reg, label='с L2-регуляризацией')
plt.xlabel('итерация')
plt.ylabel('значение функционала')
plt.title('Градиентный спуск')
plt.legend()
plt.show()
print(f"\nGD без регуляризации: MSPE на контроле = {get_func(w_gd, X_test_gd, y_test):.4f}")
print(f"GD с регуляризацией:  MSPE на контроле = {get_func(w_gd_reg, X_test_gd, y_test):.4f}")


# 23. Стохастический градиентный спуск: на каждом шаге градиент по одному случайному объекту
def sgd(X, y, w0, step_size, max_iter, eps, is_reg):
    if is_reg:
        func, grad = get_reg_func, get_reg_grad
    else:
        func, grad = get_func, get_grad

    rng = np.random.default_rng(42)
    w = w0.copy()
    history = [func(w, X, y)]
    for i in range(max_iter):
        k = rng.integers(len(y))
        w_new = w - step_size * grad(w, X[k:k + 1], y[k:k + 1])
        if i % 100 == 0:                        # функционал по всей выборке считаем раз в 100 шагов
            history.append(func(w_new, X, y))
        if np.linalg.norm(w_new - w) < eps:
            w = w_new
            break
        w = w_new
    return w, history


start = time.time()
w_sgd, history_sgd = sgd(X_train_gd, y_train, w_zeros, STEP, 20000, 1e-10, False)
time_sgd = time.time() - start
w_sgd_reg, history_sgd_reg = sgd(X_train_gd, y_train, w_zeros, STEP, 20000, 1e-10, True)

plt.figure(figsize=(9, 5))
plt.plot(history_sgd, label='без регуляризации')
plt.plot(history_sgd_reg, label='с L2-регуляризацией')
plt.xlabel('итерация / 100')
plt.ylabel('значение функционала')
plt.title('Стохастический градиентный спуск')
plt.legend()
plt.show()

# 24. Сравнение SGD и обычного градиентного спуска
start = time.time()
grad_descent(X_train_gd, y_train, w_zeros, STEP, 1000, 1e-6, False)
time_gd = time.time() - start
print(f"\nSGD: MSPE на контроле = {get_func(w_sgd, X_test_gd, y_test):.4f}, время = {time_sgd:.2f} с")
print(f"GD:  MSPE на контроле = {get_func(w_gd, X_test_gd, y_test):.4f}, время = {time_gd:.2f} с")
# Шаг SGD гораздо дешевле (один объект вместо всей выборки), но функционал убывает шумно.

# 25. Сравнение с обычной линейной регрессией (минимизирует MSE, а не MSPE)
lin_full = LinearRegression().fit(X_train, y_train)
w_lin = np.append(lin_full.coef_, lin_full.intercept_)
print(f"\nMSPE на контроле: LinearRegression = {get_func(w_lin, X_test_gd, y_test):.4f}, "
      f"градиентный спуск по MSPE = {get_func(w_gd, X_test_gd, y_test):.4f}")
# LinearRegression оптимизирует MSE, поэтому по MSPE она хуже: MSPE сильнее штрафует ошибки
# на объектах с маленькой задержкой (деление на y^2), и модель, обученная на MSPE, учитывает это.
