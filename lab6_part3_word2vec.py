"""ЛР №6, часть 3: word2vec (gensim) на отзывах IMDB.

Нужен файл с отзывами (IMDB Dataset.csv, 50k отзывов, столбец 'review').
Запуск: python lab6_part3_word2vec.py "IMDB Dataset.csv"
Если файла нет - для проверки работы используется небольшой встроенный текст (результаты будут слабые).
Установка: pip install gensim
"""
import re
import sys
import time

import pandas as pd
from gensim.models import Word2Vec
from gensim.test.utils import datapath

PATH = sys.argv[1] if len(sys.argv) > 1 else "IMDB Dataset.csv"

# ---------------------------------------------------------------------
# 1. Подготовка текста: нижний регистр, только слова, список токенов
# ---------------------------------------------------------------------
try:
    reviews = pd.read_csv(PATH)["review"].tolist()
except FileNotFoundError:
    print("Файл %s не найден - используется маленький демонстрационный текст." % PATH)
    reviews = ["the king and the queen live in the castle while the man and the woman work in the town"] * 200


def tokenize(text):
    text = re.sub(r"<[^>]+>", " ", text.lower())         # убираем html-теги (<br />)
    return re.findall(r"[a-z']+", text)


sentences = [tokenize(r) for r in reviews]
print("Отзывов:", len(sentences), "| слов всего:", sum(len(s) for s in sentences))

# ---------------------------------------------------------------------
# 2. Обучение и замер времени для разных workers
# ---------------------------------------------------------------------
for workers in [1, 4]:
    start = time.time()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=workers, epochs=5, seed=0)
    print("workers=%d: обучение %.1f с" % (workers, time.time() - start))
# (если ядер мало, ускорения может не быть)

print("Размер словаря:", len(model.wv))

# ---------------------------------------------------------------------
# 3. Примеры: похожие слова, лишнее слово, аналогии
# ---------------------------------------------------------------------


def safe(action, *args, **kwargs):
    try:
        return action(*args, **kwargs)
    except KeyError as error:                 # слова нет в словаре
        return "нет в словаре: %s" % error


for word in ["good", "movie", "actor", "boring"]:
    print("Похожие на '%s':" % word, safe(model.wv.most_similar, word, topn=5))

print("Лишнее слово в [man, woman, king, queen, car]:",
      safe(model.wv.doesnt_match, ["man", "woman", "king", "queen", "car"]))
print("king - man + woman =", safe(model.wv.most_similar, positive=["king", "woman"], negative=["man"], topn=3))

# ---------------------------------------------------------------------
# 4. Оценка на стандартном наборе аналогий Google (questions-words.txt)
# ---------------------------------------------------------------------
score, sections = model.wv.evaluate_word_analogies(datapath("questions-words.txt"))
print("Точность на аналогиях Google: %.3f" % score)
print("(на отзывах IMDB она невысока: корпус мал и тематически узок)")
