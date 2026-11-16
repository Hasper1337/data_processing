from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import boston_housing  # pip install tensorflow

# Данные из книги Шолле (раздел 3.6): цены на дома в Бостоне
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape, test_data.shape)

# Подготовка данных (3.6.2): нормализация признаков.
# Среднее и стандартное отклонение считаем ТОЛЬКО по обучающим данным
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Обучаем Ridge
model = Ridge()
model.fit(train_data, train_targets)

# Среднеквадратичная ошибка
predictions = model.predict(test_data)
print("MSE:", mean_squared_error(test_targets, predictions))
print("R^2:", model.score(test_data, test_targets))
