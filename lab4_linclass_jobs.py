"""ЛР №4: линейная классификация - предсказание зарплаты по тексту вакансии.

Данные: DataAnalyst.csv (Kaggle, data analyst jobs).
Запуск: python lab4_linclass_jobs.py DataAnalyst.csv
Пункты 1-16 из блокнота homework-practice-02-linclass.
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder

PATH = sys.argv[1] if len(sys.argv) > 1 else "DataAnalyst.csv"

# =====================================================================
# Подготовка данных (как в блокноте)
# =====================================================================
df = pd.read_csv(PATH)
df = df[df["Salary Estimate"] != "-1"].copy()


def average_salary(text):
    """'$37K-$66K (Glassdoor est.)' -> 51500. (В блокноте считается разность границ,
    но по смыслу задачи нужна именно оценка зарплаты, поэтому берём среднее.)"""
    low, high = [int(v) for v in re.findall(r"\d+", text)[:2]]
    return (low + high) / 2 * 1000


df["Salary Estimate"] = df["Salary Estimate"].apply(average_salary)
median = df["Salary Estimate"].median()
df["Salary Estimate"] = (df["Salary Estimate"] > median).astype(int)   # 1 - высокая зарплата
df = df.drop(columns=["Revenue", "Easy Apply", "Rating", "Unnamed: 0"], errors="ignore")
print("Объектов:", len(df), "| доля класса 1:", df["Salary Estimate"].mean().round(3))

TARGET = "Salary Estimate"
TEXT_COLS = ["Job Title", "Job Description"]
CAT_COLS = [c for c in df.columns if c not in TEXT_COLS + [TARGET]]


def evaluate(name, y_true, proba):
    """ROC-AUC, F-мера и матрица ошибок."""
    pred = (proba > 0.5).astype(int)
    print("%s: ROC-AUC=%.4f F1=%.4f accuracy=%.4f" % (
        name, roc_auc_score(y_true, proba), f1_score(y_true, pred), accuracy_score(y_true, pred)))
    print(confusion_matrix(y_true, pred))


def fit_lr(X, y, C=1.0):
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)


# =====================================================================
# 1. Разбиение 70/30
# =====================================================================
train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
y_train, y_test = train[TARGET].values, test[TARGET].values
print("1. train:", len(train), "test:", len(test))

# =====================================================================
# 2-5. Все признаки склеены в один текст + CountVectorizer
# =====================================================================


def all_text(data):
    return data.drop(columns=[TARGET]).astype(str).agg(" ".join, axis=1)


txt_train, txt_test = all_text(train), all_text(test)

cv = CountVectorizer()
Xc_train = cv.fit_transform(txt_train)
Xc_test = cv.transform(txt_test)
print("2. размер словаря CountVectorizer:", Xc_train.shape[1])

lr_cv = fit_lr(Xc_train, y_train)
print("3-4. логистическая регрессия на мешке слов")
evaluate("CountVectorizer", y_test, lr_cv.predict_proba(Xc_test)[:, 1])


def show_top_words(model, vectorizer, n=10):
    words = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    print("   сильнее всего в сторону НИЗКОЙ зарплаты:", list(words[order[:n]]))
    print("   сильнее всего в сторону ВЫСОКОЙ зарплаты:", list(words[order[-n:]][::-1]))


print("5. веса признаков")
show_top_words(lr_cv, cv)

# =====================================================================
# 6-10. TF-IDF на склеенном тексте
# =====================================================================
tf = TfidfVectorizer()
Xt_train = tf.fit_transform(txt_train)
Xt_test = tf.transform(txt_test)
lr_tf = fit_lr(Xt_train, y_train)
print("6-8. логистическая регрессия на tf-idf")
evaluate("TF-IDF", y_test, lr_tf.predict_proba(Xt_test)[:, 1])
print("9. сравнение с п. 5: смотрите ROC-AUC/F1 выше (CountVectorizer и TF-IDF).")
print("10. веса признаков")
show_top_words(lr_tf, tf)

# =====================================================================
# 11. Текст только из Job Title + Job Description, остальное - категориальные
# =====================================================================


def job_text(data):
    return data[TEXT_COLS].astype(str).agg(" ".join, axis=1)


tf2 = TfidfVectorizer()
T_train = tf2.fit_transform(job_text(train))
T_test = tf2.transform(job_text(test))
print("11. размер tf-idf по Job Title + Job Description:", T_train.shape)

# =====================================================================
# 12. One-hot для категориальных признаков
# =====================================================================
ohe = OneHotEncoder(handle_unknown="ignore")
O_train = ohe.fit_transform(train[CAT_COLS].astype(str))
O_test = ohe.transform(test[CAT_COLS].astype(str))
X12_train = hstack([T_train, O_train]).tocsr()
X12_test = hstack([T_test, O_test]).tocsr()
lr12 = fit_lr(X12_train, y_train)
print("12. tf-idf + one-hot")
res12 = roc_auc_score(y_test, lr12.predict_proba(X12_test)[:, 1])
evaluate("one-hot", y_test, lr12.predict_proba(X12_test)[:, 1])

# =====================================================================
# 13. Счётчики (target encoding без утечки)
# =====================================================================


def counters_loo(train_df, y, columns):
    """Для обучающей выборки: доля класса 1 среди ДРУГИХ объектов с тем же значением
    признака (текущий объект исключается). Считается через groupby без циклов по объектам."""
    prior = y.mean()
    out = np.zeros((len(train_df), len(columns)))
    ys = pd.Series(y, index=train_df.index)
    for j, col in enumerate(columns):
        g = train_df[col].astype(str)
        s = ys.groupby(g).transform("sum")
        c = ys.groupby(g).transform("count")
        loo = (s - ys) / (c - 1)
        out[:, j] = loo.where(c > 1, prior).values      # редкие значения -> общая доля
    return out


def counters_apply(train_df, y, other_df, columns):
    """Для контрольной выборки: доля класса 1 по всей обучающей выборке."""
    prior = y.mean()
    out = np.zeros((len(other_df), len(columns)))
    ys = pd.Series(y, index=train_df.index)
    for j, col in enumerate(columns):
        stats = ys.groupby(train_df[col].astype(str)).mean()
        out[:, j] = other_df[col].astype(str).map(stats).fillna(prior).values
    return out


C_train = counters_loo(train, y_train, CAT_COLS)
C_test = counters_apply(train, y_train, test, CAT_COLS)
X13_train = hstack([T_train, csr_matrix(C_train)]).tocsr()
X13_test = hstack([T_test, csr_matrix(C_test)]).tocsr()
lr13 = fit_lr(X13_train, y_train)
print("13. tf-idf + счётчики")
evaluate("счётчики", y_test, lr13.predict_proba(X13_test)[:, 1])
print("14. сравните п. 12 (one-hot) и п. 13 (счётчики) по метрикам выше:")
print("    счётчики дают по одному признаку на столбец (компактно), one-hot - тысячи столбцов.")

# =====================================================================
# 15-16. Подбор числа фолдов для счётчиков и коэффициента регуляризации C
# =====================================================================


def counters_kfold(train_df, y, columns, n_folds, seed=0):
    """Счётчики для обучающей выборки по K фолдам: значение для объекта считается по остальным фолдам."""
    prior = y.mean()
    out = np.full((len(train_df), len(columns)), prior)
    ys = pd.Series(y, index=train_df.index)
    for fit_idx, apply_idx in KFold(n_folds, shuffle=True, random_state=seed).split(train_df):
        for j, col in enumerate(columns):
            stats = ys.iloc[fit_idx].groupby(train_df[col].astype(str).iloc[fit_idx]).mean()
            values = train_df[col].astype(str).iloc[apply_idx].map(stats).fillna(prior)
            out[apply_idx, j] = values.values
    return out


sub_train, valid = train_test_split(train, test_size=0.2, random_state=42)
y_sub, y_valid = sub_train[TARGET].values, valid[TARGET].values
T_sub = tf2.fit_transform(job_text(sub_train))
T_valid = tf2.transform(job_text(valid))
V_valid = counters_apply(sub_train, y_sub, valid, CAT_COLS)

best = (-1, None, None)
print("15. подбор (число фолдов, C) по accuracy на валидации:")
for n_folds in [2, 3, 5, 10]:
    C_sub = counters_kfold(sub_train, y_sub, CAT_COLS, n_folds)
    X_sub = hstack([T_sub, csr_matrix(C_sub)]).tocsr()
    X_val = hstack([T_valid, csr_matrix(V_valid)]).tocsr()
    for C in [0.01, 0.1, 1, 10, 100]:
        acc = accuracy_score(y_valid, fit_lr(X_sub, y_sub, C).predict(X_val))
        print("    фолдов=%-2d C=%-6g accuracy=%.4f" % (n_folds, C, acc))
        if acc > best[0]:
            best = (acc, n_folds, C)
print("    лучшее: accuracy=%.4f, фолдов=%d, C=%g" % best)

# 16. Итоговая модель на всей обучающей выборке
_, best_folds, best_C = best
tf2 = TfidfVectorizer()
T_train = tf2.fit_transform(job_text(train))
T_test = tf2.transform(job_text(test))
C_train = counters_kfold(train, y_train, CAT_COLS, best_folds)
C_test = counters_apply(train, y_train, test, CAT_COLS)
X_train = hstack([T_train, csr_matrix(C_train)]).tocsr()
X_test = hstack([T_test, csr_matrix(C_test)]).tocsr()
final = fit_lr(X_train, y_train, best_C)
print("16. итоговая модель с подобранными гиперпараметрами")
evaluate("итог", y_test, final.predict_proba(X_test)[:, 1])
