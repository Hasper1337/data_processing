# -*- coding: utf-8 -*-
"""
Лабораторная работа №2. Вариант 1: поиск гипотез методом Sliding Window.

Пайплайн:
  1. Ground truth (реальные объекты на изображении) находится каскадом Хаара
     (haarcascade_frontalcatface.xml, встроен в opencv-python) — используется
     как "эталон" вместо ручной разметки.
  2. Sliding window перебирает все прямоугольники разного размера/масштаба.
  3. Эвристический отсев (objectness): гипотезы с низкой плотностью границ
     (Canny) отбрасываются сразу как Negative и НЕ идут на классификацию.
  4. Классификация оставшихся гипотез: тот же тип классификатора
     (каскад Хаара), но применяется отдельно к каждому вырезанному окну —
     это моделирует шаг "R-CNN-подобной" классификации гипотез.
  5. Итоговый Predicted = Positive только если гипотеза прошла эвристику
     И классификатор подтвердил наличие объекта.
  6. Actual (реальное присутствие объекта в окне) определяется через IoU
     окна с ground-truth рамкой (порог IOU_THRESHOLD).
  7. Считается таблица TP/TN/FP/FN и Accuracy = (TP+TN)/(P+N).

Использование:
    python3 sliding_window_lab.py --images_dir demo_images --out_dir output
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- параметры

WINDOW_SIZES = [(80, 80), (120, 120), (160, 160), (120, 160)]  # (w, h), несколько масштабов/пропорций
STEP_SIZE = 40                # шаг скользящего окна в пикселях
EDGE_DENSITY_THRESHOLD = 0.04 # порог эвристики objectness (доля пикселей-границ в окне)
IOU_THRESHOLD = 0.3           # порог IoU для признания окна "actual positive"

CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface.xml")
_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ------------------------------------------------------------- базовые шаги

def get_ground_truth_boxes(gray_image):
    """
    'Эталонные' рамки объекта. В отсутствие ручной разметки используем
    каскад Хаара, применённый ко всему изображению, как источник ground truth.
    Возвращает список (x1, y1, x2, y2).
    """
    detections = _cascade.detectMultiScale(
        gray_image, scaleFactor=1.02, minNeighbors=2, minSize=(40, 40)
    )
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in detections]
    return boxes


def sliding_window_proposals(image_shape, window_sizes=WINDOW_SIZES, step=STEP_SIZE):
    """Генератор всех прямоугольников-гипотез (x1, y1, x2, y2)."""
    H, W = image_shape[:2]
    for (w, h) in window_sizes:
        if w > W or h > H:
            continue
        for y in range(0, H - h + 1, step):
            for x in range(0, W - w + 1, step):
                yield (x, y, x + w, y + h)


def edge_density(gray_crop):
    """Доля пикселей-границ (Canny) в вырезанном окне — простая мера objectness."""
    edges = cv2.Canny(gray_crop, 80, 160)
    return float(np.count_nonzero(edges)) / edges.size


def classify_window(gray_crop):
    """
    Классификатор гипотезы: применяет каскад Хаара к самому окну.
    True, если внутри окна найден объект искомого класса.
    """
    if gray_crop.shape[0] < 24 or gray_crop.shape[1] < 24:
        return False
    dets = _cascade.detectMultiScale(
        gray_crop, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
    )
    return len(dets) > 0


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def max_iou_with_gt(box, gt_boxes):
    if not gt_boxes:
        return 0.0
    return max(iou(box, gt) for gt in gt_boxes)


# ------------------------------------------------------------- обработка 1 изображения

def process_image(path, out_dir=None, save_vis=True):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gt_boxes = get_ground_truth_boxes(gray)

    counts = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    n_heuristic_positive = 0   # гипотезы, прошедшие эвристику (PP на шаге отбора)
    n_heuristic_negative = 0   # отсеянные эвристикой (сразу Negative)
    n_classified_positive = 0  # из прошедших эвристику подтверждено классификатором

    vis = image.copy()
    for gt in gt_boxes:
        cv2.rectangle(vis, gt[:2], gt[2:], (0, 255, 0), 2)  # GT — зелёным

    for box in sliding_window_proposals(image.shape):
        x1, y1, x2, y2 = box
        crop_gray = gray[y1:y2, x1:x2]

        passed_heuristic = edge_density(crop_gray) > EDGE_DENSITY_THRESHOLD

        if passed_heuristic:
            n_heuristic_positive += 1
            predicted_positive = classify_window(crop_gray)
            if predicted_positive:
                n_classified_positive += 1
        else:
            n_heuristic_negative += 1
            predicted_positive = False  # отсеяно эвристикой -> Predicted Negative

        actual_positive = max_iou_with_gt(box, gt_boxes) >= IOU_THRESHOLD

        if predicted_positive and actual_positive:
            counts["TP"] += 1
            color = (0, 0, 255)      # красный — TP
        elif predicted_positive and not actual_positive:
            counts["FP"] += 1
            color = (0, 165, 255)    # оранжевый — FP
        elif (not predicted_positive) and actual_positive:
            counts["FN"] += 1
            color = (255, 0, 255)    # розовый — FN
        else:
            counts["TN"] += 1
            color = None             # TN не рисуем, иначе всё изображение будет в рамках

        if save_vis and color is not None:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

    total = sum(counts.values())
    accuracy = (counts["TP"] + counts["TN"]) / total if total else 0.0

    if save_vis and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "vis_" + os.path.basename(path))
        cv2.imwrite(out_path, vis)

    return {
        "image": os.path.basename(path),
        "gt_boxes": len(gt_boxes),
        "proposals_total": total,
        "heuristic_positive": n_heuristic_positive,
        "heuristic_negative": n_heuristic_negative,
        "classified_positive": n_classified_positive,
        **counts,
        "accuracy": accuracy,
    }


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Sliding window object proposal lab")
    ap.add_argument("--images_dir", default="demo_images", help="папка с изображениями (10 картинок по заданию)")
    ap.add_argument("--out_dir", default="output", help="куда сохранять визуализации и отчёт")
    args = ap.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.png"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
    )
    if not paths:
        print(f"Нет изображений в {args.images_dir}")
        return

    results = []
    for p in paths:
        r = process_image(p, out_dir=args.out_dir)
        results.append(r)
        print(
            f"{r['image']:20s} | GT={r['gt_boxes']} | proposals={r['proposals_total']:4d} "
            f"(PP_heur={r['heuristic_positive']:3d}, PN_heur={r['heuristic_negative']:3d}, "
            f"classified+={r['classified_positive']:3d}) "
            f"| TP={r['TP']:3d} FN={r['FN']:3d} FP={r['FP']:3d} TN={r['TN']:3d} "
            f"| Accuracy={r['accuracy']:.4f}"
        )

    agg = {k: sum(r[k] for r in results) for k in ["TP", "FN", "FP", "TN"]}
    total = sum(agg.values())
    agg_accuracy = (agg["TP"] + agg["TN"]) / total if total else 0.0

    print("\n=== Итоговая таблица (сумма по всем изображениям) ===")
    print(f"{'':15s}{'Predicted Positive':>20s}{'Predicted Negative':>20s}")
    print(f"{'Actual Positive':15s}{agg['TP']:>20d}{agg['FN']:>20d}")
    print(f"{'Actual Negative':15s}{agg['FP']:>20d}{agg['TN']:>20d}")
    print(f"\nAccuracy = (TP+TN)/(P+N) = {agg_accuracy:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"per_image": results, "aggregate": {**agg, "accuracy": agg_accuracy}}, f,
                   ensure_ascii=False, indent=2)
    print(f"\nПодробный отчёт: {os.path.join(args.out_dir, 'report.json')}")
    print(f"Визуализации окон: {args.out_dir}/vis_*.jpg (красный=TP, оранжевый=FP, розовый=FN)")


if __name__ == "__main__":
    main()
