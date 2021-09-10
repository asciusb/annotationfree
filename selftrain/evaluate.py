# Evaluate character detection.

import argparse
import csv
import os
import os.path as path
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
import tqdm


def load_tsv(file_tsv, factor=1):
    with open(file_tsv, "r") as fd:
        reader = csv.reader(
            fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
        )
        boxes = []
        for line in reader:
            box = {}
            box["className"] = line[0]
            box["xStart"] = int(line[1]) * factor
            box["yStart"] = int(line[2]) * factor
            box["xEnd"] = int(line[3]) * factor
            box["yEnd"] = int(line[4]) * factor
            boxes.append(box)
    return boxes


def evaluate(file_image, file_sys, file_gt, file_font, file_eval_png, file_eval_txt, detection_size):

    # Rescale if necessary

    source_img = Image.open(file_image).convert("RGBA")
    factor = 1
    if detection_size > 0:
        width, height = source_img.size
        max_size = max(width, height)
        if max_size != detection_size:
            factor = max_size / detection_size

    # Read shapes

    bboxes_sys = load_tsv(file_sys, factor)
    bboxes_gt = load_tsv(file_gt)
    shapes_sys = []
    for bbox in bboxes_sys:
        shape = box(bbox["xStart"], bbox["yStart"], bbox["xEnd"], bbox["yEnd"])
        shapes_sys.append(shape)
    shapes_gt = []
    for bbox in bboxes_gt:
        shape = box(bbox["xStart"], bbox["yStart"], bbox["xEnd"], bbox["yEnd"])
        shapes_gt.append(shape)

    # Calculate LSAP

    n_gt = len(shapes_gt)
    n_sys = len(shapes_sys)
    n = n_gt + n_sys
    cost = np.ones((n, n))
    for i in range(0, n):
        for j in range(0, n):
            cost[i][j] = 2.0
    for i in range(0, n_gt):
        for j in range(0, n_sys):
            intersection = shapes_gt[i].intersection(shapes_sys[j])
            union = shapes_gt[i].union(shapes_sys[j])
            iou = intersection.area / union.area
            cost[i][j] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)
    matchings = []
    deletions = []
    insertions = []
    for i in range(0, n):
        j = col_ind[i]
        if i < n_gt:
            if j < n_sys:
                if cost[i][j] < 1.0:
                    matchings.append([shapes_gt[i], shapes_sys[j], bboxes_gt[i], bboxes_sys[j]])
                else:
                    deletions.append(shapes_gt[i])
                    insertions.append(shapes_sys[j])
            else:
                deletions.append(shapes_gt[i])
        else:
            if j < n_sys:
                insertions.append(shapes_sys[j])

    # Detection IOU = average over all matchings, deletions, and insertions

    iou = 0.0
    iou_samples = 0.0
    for m in matchings:
        iou += m[0].intersection(m[1]).area / m[0].union(m[1]).area
        iou_samples += 1
    iou_samples += len(deletions)
    iou_samples += len(insertions)
    detection_iou = iou / iou_samples

    # Detection Accuracy = (N_gt - Err_s - Err_d - Err_i) / N_gt

    threshold = 0.5
    num_gt = len(shapes_gt)
    num_ok = 0
    err_s = 0
    err_d = len(deletions)
    err_i = len(insertions)
    for m in matchings:
        iou = m[0].intersection(m[1]).area / m[0].union(m[1]).area
        if iou < threshold:
            err_s += 1
        else:
            num_ok += 1
    detection_acc = (num_gt - err_s - err_d - err_i) / float(num_gt)

    # Precision, Recall, F1

    precision = num_ok / n_sys
    recall = num_ok / n_gt
    f1score = 2 * precision * recall / (precision + recall)

    # Classification Accuracy = (N_gt - Err_s - Err_d - Err_i) / N_gt

    cls_num_ok = 0
    cls_err_s = 0
    cls_err_d = len(deletions)
    cls_err_i = len(insertions)
    for m in matchings:
        if m[2]["className"] != m[3]["className"]:
            cls_err_s += 1
        else:
            cls_num_ok += 1
    classification_acc = (num_gt - cls_err_s - cls_err_d - cls_err_i) / float(num_gt)

    # Save evaluations

    with open(file_eval_txt, "w") as f:
        f.write("Detection-GT: {}, OK: {}, S: {}, D: {}, I: {}\n".format(num_gt, num_ok, err_s, err_d, err_i))
        f.write("Detection-Accuracy: {}\n".format(detection_acc))
        f.write("Detection-IOU: {}\n".format(detection_iou))
        f.write("Classification-GT: {}, OK: {}, S: {}, D: {}, I: {}\n".format(num_gt, cls_num_ok, cls_err_s, cls_err_d, cls_err_i))
        f.write("Classification-Accuracy: {}\n".format(classification_acc))
        f.write("Precison-Recall-F1score: {}, {}, {}\n".format(precision, recall, f1score))

    # Draw results

    font = ImageFont.truetype(file_font, 24)
    draw = ImageDraw.Draw(source_img)
    for b in deletions:
        draw.rectangle(((b.bounds[0], b.bounds[1]), (b.bounds[2], b.bounds[3])), fill=None, width=3, outline="red")
    for b in insertions:
        draw.rectangle(((b.bounds[0], b.bounds[1]), (b.bounds[2], b.bounds[3])), fill=None, width=3, outline="magenta")
    for m in matchings:
        b = m[0]
        draw.rectangle(((b.bounds[0], b.bounds[1]), (b.bounds[2], b.bounds[3])), fill=None, outline="blue")
        b = m[1]
        if m[2]["className"] == m[3]["className"]:
            draw.rectangle(((b.bounds[0], b.bounds[1]), (b.bounds[2], b.bounds[3])), fill=None, width=3, outline="cyan")
        else:
            draw.rectangle(((b.bounds[0], b.bounds[1]), (b.bounds[2], b.bounds[3])), fill=None, width=3, outline="green")

    source_img.save(file_eval_png, "PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # required

    parser.add_argument(
        "--dir-images",
        dest="dir_images",
        required=True,
        type=str,
        help="Images root directory",
    )
    parser.add_argument(
        "--image-type",
        dest="image_type",
        required=True,
        type=str,
        help="Image type (jpg, png, ...)",
    )
    parser.add_argument(
        "--dir-detection",
        dest="dir_detection",
        required=True,
        type=str,
        help="Directory to load detection results",
    )
    parser.add_argument(
        "--dir-evaluation",
        dest="dir_evaluation",
        required=True,
        type=str,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--font",
        dest="font_ttf",
        required=True,
        type=str,
        help="Font (ttf)",
    )

    # optional

    parser.add_argument(
        "--detection-size",
        dest="detection_size",
        required=False,
        default=0,
        type=int,
        help="Detection size (if different from original image size)",
    )

    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    img_dir = options.dir_images
    img_type_lower = options.image_type.lower()
    img_type_upper = options.image_type.upper()
    dir_gt = options.dir_images
    dir_detection = options.dir_detection
    dir_evaluation = options.dir_evaluation
    dir_evaluation_view = path.join(dir_evaluation, "view")
    file_font = options.font_ttf
    detection_size = options.detection_size

    if not path.exists(dir_evaluation):
        os.makedirs(dir_evaluation)
    if not path.exists(dir_evaluation_view):
        os.makedirs(dir_evaluation_view)

    print("Begin evaluation..")
    files = sorted(os.listdir(img_dir))
    log_page = tqdm.tqdm(total=len(files), desc='Evaluate', position=1)
    for filename in files:
        if filename.endswith(".{}".format(img_type_lower)) or filename.endswith(".{}".format(img_type_upper)):
            file_image = path.join(img_dir, filename)
            file_sys = path.join(dir_detection, "{}.tsv".format(filename.split(".")[0]))
            file_gt = path.join(dir_gt, "{}.tsv".format(filename.split(".")[0]))
            file_eval_png = path.join(dir_evaluation_view, "{}.png".format(filename.split(".")[0]))
            file_eval_txt = path.join(dir_evaluation, "{}.txt".format(filename.split(".")[0]))
            evaluate(file_image, file_sys, file_gt, file_font, file_eval_png, file_eval_txt, detection_size)
        log_page.update(1)

    print(".. done.")
