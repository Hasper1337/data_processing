import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder

jobs = pd.read_csv("DataAnalyst.csv")

# Подготовка данных
# Убираем вакансии без оценки зарплаты
jobs = jobs[jobs['Salary Estimate'] != '-1'].copy()


# '$37K-$66K (Glassdoor est.)' -> 51500.0 (середина вилки)
# (в блокноте берётся разность границ, но для оценки зарплаты логичнее середина)
def parse_salary(text):
    numbers = re.findall(r'\d+', text)
    low, high = int(numbers[0]), int(numbers[1])
    return (low + high) / 2 * 1000


jobs['Salary Estimate'] = jobs['Salary Estimate'].apply(parse_salary)

# Целевая переменная: 1 - зарплата выше медианы, 0 - не выше
median_salary = jobs['Salary Estimate'].median()
jobs['Salary Estimate'] = (jobs['Salary Estimate'] > median_salary).astype(int)

jobs = jobs.drop(columns=['Unnamed: 0', 'Rating', 'Revenue', 'Easy Apply'], errors='ignore')
# Пропуски заменяем на '-1' (так в этом датасете уже обозначены пустые значения), иначе склейка текста падает
jobs = jobs.fillna('-1')
print(f"Вакансий: {len(jobs)}, медиана зарплаты: {median_salary:.0f}")
print(f"Доля класса 1: {jobs['Salary Estimate'].mean():.3f}")

target = 'Salary Estimate'
text_cols = ['Job Title', 'Job Description']
cat_cols = [col for col in jobs.columns if col not in text_cols + [target]]


# Одна функция для вывода метрик, чтобы не повторять код
def print_metrics(name, y_true, proba):
    pred = (proba > 0.5).astype(int)
    print(f"\n{name}:")
    print(f"ROC-AUC = {roc_auc_score(y_true, proba):.4f}, F1 = {f1_score(y_true, pred):.4f}, "
          f"accuracy = {accuracy_score(y_true, pred):.4f}")
    print(f"Матрица ошибок:\n{confusion_matrix(y_true, pred)}")


# 1. Разбиение на обучающую и тестовую выборки 70/30
train, test = train_test_split(jobs, test_size=0.3, shuffle=True, random_state=42)
y_train, y_test = train[target].values, test[target].values
print(f"\nОбучение: {train.shape}, тест: {test.shape}")

# 2. Склеиваем все признаки в один текст и строим мешок слов (CountVectorizer)
text_train = train.drop(columns=[target]).astype(str).agg(' '.join, axis=1)
text_test = test.drop(columns=[target]).astype(str).agg(' '.join, axis=1)

count_vec = CountVectorizer()
X_train_count = count_vec.fit_transform(text_train)
X_test_count = count_vec.transform(text_test)
print(f"Размер словаря: {X_train_count.shape[1]}")

# 3-4. Логистическая регрессия на мешке слов
log_reg = LogisticRegression(max_iter=2000).fit(X_train_count, y_train)
print_metrics("CountVectorizer + LogisticRegression", y_test, log_reg.predict_proba(X_test_count)[:, 1])

# 5. Слова с самыми большими положительными и отрицательными весами
words = np.array(count_vec.get_feature_names_out())
order = np.argsort(log_reg.coef_[0])
print(f"\nСлова в сторону низкой зарплаты: {list(words[order[:10]])}")
print(f"Слова в сторону высокой зарплаты: {list(words[order[-10:][::-1]])}")

# 6-8. То же самое, но с TF-IDF вместо мешка слов
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(text_train)
X_test_tfidf = tfidf_vec.transform(text_test)

log_reg_tfidf = LogisticRegression(max_iter=2000).fit(X_train_tfidf, y_train)
print_metrics("TF-IDF + LogisticRegression", y_test, log_reg_tfidf.predict_proba(X_test_tfidf)[:, 1])

# 9. Сравнение: смотрим на ROC-AUC и F1 у CountVectorizer и TF-IDF выше.
# TF-IDF уменьшает вес слов, которые встречаются почти во всех вакансиях.

# 10. Веса слов для TF-IDF
words = np.array(tfidf_vec.get_feature_names_out())
order = np.argsort(log_reg_tfidf.coef_[0])
print(f"\nСлова в сторону низкой зарплаты: {list(words[order[:10]])}")
print(f"Слова в сторону высокой зарплаты: {list(words[order[-10:][::-1]])}")

# 11. TF-IDF только по тексту (Job Title + Job Description), остальные признаки - категориальные
job_text_train = train[text_cols].astype(str).agg(' '.join, axis=1)
job_text_test = test[text_cols].astype(str).agg(' '.join, axis=1)

tfidf_job = TfidfVectorizer()
T_train = tfidf_job.fit_transform(job_text_train)
T_test = tfidf_job.transform(job_text_test)
print(f"\nTF-IDF по Job Title + Job Description: {T_train.shape}")
print(f"Категориальные признаки: {cat_cols}")

# 12. One-hot кодирование категориальных признаков + TF-IDF текста
one_hot = OneHotEncoder(handle_unknown='ignore')
O_train = one_hot.fit_transform(train[cat_cols].astype(str))
O_test = one_hot.transform(test[cat_cols].astype(str))

X_train_12 = hstack([T_train, O_train]).tocsr()
X_test_12 = hstack([T_test, O_test]).tocsr()
log_reg_12 = LogisticRegression(max_iter=2000).fit(X_train_12, y_train)
print_metrics("TF-IDF + one-hot", y_test, log_reg_12.predict_proba(X_test_12)[:, 1])

# 13. Счётчики: категорию заменяем на долю вакансий с высокой зарплатой среди вакансий с той же категорией.
# Чтобы не было утечки ответа, на обучении текущий объект в подсчёт не включаем (leave-one-out),
# а на тесте используем статистику по всей обучающей выборке.
prior = y_train.mean()   # общая доля класса 1 - для редких и новых категорий
y_train_series = pd.Series(y_train, index=train.index)

C_train = np.zeros((len(train), len(cat_cols)))
C_test = np.zeros((len(test), len(cat_cols)))
for j, col in enumerate(cat_cols):
    values_train = train[col].astype(str)
    values_test = test[col].astype(str)

    sum_in_group = y_train_series.groupby(values_train).transform('sum')
    count_in_group = y_train_series.groupby(values_train).transform('count')
    loo = (sum_in_group - y_train_series) / (count_in_group - 1)
    C_train[:, j] = loo.where(count_in_group > 1, prior).values

    group_mean = y_train_series.groupby(values_train).mean()
    C_test[:, j] = values_test.map(group_mean).fillna(prior).values

X_train_13 = hstack([T_train, csr_matrix(C_train)]).tocsr()
X_test_13 = hstack([T_test, csr_matrix(C_test)]).tocsr()
log_reg_13 = LogisticRegression(max_iter=2000).fit(X_train_13, y_train)
print_metrics("TF-IDF + счётчики", y_test, log_reg_13.predict_proba(X_test_13)[:, 1])

# 14. Сравнение one-hot и счётчиков: смотрим на метрики п. 12 и п. 13.
# Счётчики дают всего по одному столбцу на признак, а one-hot - по столбцу на каждую категорию.


# 15. Счётчики по фолдам: для объектов одного фолда считаем статистику по остальным фолдам.
# Подбираем число фолдов и коэффициент регуляризации C по accuracy на валидационной выборке.
def kfold_counters(data, y, n_folds):
    y_series = pd.Series(y, index=data.index)
    prior = y.mean()
    result = np.full((len(data), len(cat_cols)), prior)
    for fit_idx, apply_idx in KFold(n_folds, shuffle=True, random_state=0).split(data):
        for j, col in enumerate(cat_cols):
            values = data[col].astype(str)
            group_mean = y_series.iloc[fit_idx].groupby(values.iloc[fit_idx]).mean()
            result[apply_idx, j] = values.iloc[apply_idx].map(group_mean).fillna(prior).values
    return result


def test_counters(train_data, y, other_data):
    y_series = pd.Series(y, index=train_data.index)
    result = np.zeros((len(other_data), len(cat_cols)))
    for j, col in enumerate(cat_cols):
        group_mean = y_series.groupby(train_data[col].astype(str)).mean()
        result[:, j] = other_data[col].astype(str).map(group_mean).fillna(y.mean()).values
    return result


sub_train, valid = train_test_split(train, test_size=0.2, random_state=42)
y_sub, y_valid = sub_train[target].values, valid[target].values

tfidf_sub = TfidfVectorizer()
T_sub = tfidf_sub.fit_transform(sub_train[text_cols].astype(str).agg(' '.join, axis=1))
T_valid = tfidf_sub.transform(valid[text_cols].astype(str).agg(' '.join, axis=1))
X_valid = hstack([T_valid, csr_matrix(test_counters(sub_train, y_sub, valid))]).tocsr()

best_acc, best_folds, best_C = 0, None, None
print("\nПодбор числа фолдов и C:")
for n_folds in [2, 3, 5, 10]:
    X_sub = hstack([T_sub, csr_matrix(kfold_counters(sub_train, y_sub, n_folds))]).tocsr()
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=2000).fit(X_sub, y_sub)
        acc = accuracy_score(y_valid, model.predict(X_valid))
        print(f"фолдов = {n_folds}, C = {C}: accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc, best_folds, best_C = acc, n_folds, C
print(f"Лучшее: accuracy = {best_acc:.4f}, фолдов = {best_folds}, C = {best_C}")

# 16. Итоговая модель с лучшими параметрами на всей обучающей выборке
X_train_16 = hstack([T_train, csr_matrix(kfold_counters(train, y_train, best_folds))]).tocsr()
X_test_16 = hstack([T_test, csr_matrix(test_counters(train, y_train, test))]).tocsr()
final_model = LogisticRegression(C=best_C, max_iter=2000).fit(X_train_16, y_train)
print_metrics("Итоговая модель", y_test, final_model.predict_proba(X_test_16)[:, 1])
