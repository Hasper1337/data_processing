import os
import re
from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Путь к распакованной папке data.zip (внутри должны быть папки neg и pos)
DATA_DIR = "data"
VOCAB_SIZE = 10000  # размер словаря (длина вектора)

# Простой стемминг (приведение слов к единому виду). Если nltk не установлен - пропускаем.
try:
    from nltk.stem import PorterStemmer
    stem = PorterStemmer().stem
except ImportError:
    def stem(word):
        return word


def to_tokens(text):
    """Текст -> список слов в нижнем регистре, приведённых к основе."""
    words = re.findall(r"[a-z']+", text.lower())
    return [stem(w) for w in words]


# 1. Читаем отзывы: neg -> 0, pos -> 1
texts = []
emotion = []
for label, folder in ((0, "neg"), (1, "pos")):
    folder_path = os.path.join(DATA_DIR, folder)
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name), encoding="utf-8") as f:
            texts.append(f.read())
            emotion.append(label)
emotion = np.array(emotion)

# 2. Токенизация и словарь из самых частых слов
all_tokens = [to_tokens(t) for t in texts]
counter = Counter(w for tokens in all_tokens for w in tokens)
vocab = [w for w, _ in counter.most_common(VOCAB_SIZE)]
word_index = {w: i for i, w in enumerate(vocab)}

# 3. Прямое кодирование: 1 - слово есть в отзыве, 0 - нет
train_data_features = np.zeros((len(texts), len(vocab)), dtype=np.uint8)
for i, tokens in enumerate(all_tokens):
    for w in tokens:
        if w in word_index:
            train_data_features[i, word_index[w]] = 1

# 4. Разбиваем на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(
    train_data_features, emotion, test_size=0.2, random_state=0)

# 5. Обучаем классификатор и проверяем точность
clf = BernoulliNB()
clf.fit(X_train, Y_train)
print("Accuracy:", metrics.accuracy_score(clf.predict(X_test), Y_test))
