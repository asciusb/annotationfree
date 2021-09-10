# Cluster detected characters to columns and save column characters.

import argparse
import csv
import os
import os.path as path
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image, ImageDraw
import tqdm


def read_boxes(path_boxes, factor=1):
    with open(path_boxes, "r") as fd:
        reader = csv.reader(
            fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
        )
        boxes = []
        for bbox in reader:
            rescale_bbox(bbox, factor)
            box = Box(bbox)
            boxes.append(box)
    return boxes


def rescale_bbox(bbox, factor):
    if factor != 1:
        bbox[1] = int(int(bbox[1]) * factor)
        bbox[2] = int(int(bbox[2]) * factor)
        bbox[3] = int(int(bbox[3]) * factor)
        bbox[4] = int(int(bbox[4]) * factor)


class Box:

    def __init__(self, bbox):
        self.className = bbox[0]
        self.xStart = int(bbox[1])
        self.yStart = int(bbox[2])
        self.xEnd = int(bbox[3])
        self.yEnd = int(bbox[4])
        self.xCenter = (self.xStart + self.xEnd) / 2
        self.yCenter = (self.yStart + self.yEnd) / 2
        self.width = self.xEnd - self.xStart
        self.height = self.yEnd - self.yStart


def cluster_columns(boxes, median_box, max_center_deviation, eps, min_samples):
    centers = []
    for box in boxes:
        centers.append([box.xCenter, box.yCenter, median_box.width, median_box.height, max_center_deviation])

    def column_distance(box1, box2):
        inf = 100000
        xCenter1 = box1[0]
        xCenter2 = box2[0]
        yCenter1 = box1[1]
        yCenter2 = box2[1]
        column_width = box1[2]
        character_height = box1[3]
        max_center_deviation = box1[4]
        if abs(xCenter1 - xCenter2) / column_width > max_center_deviation:
            return inf
        return abs(yCenter1 - yCenter2) / character_height

    X = np.array(centers)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=column_distance).fit(X)

    boxes_column_ok = []
    outliers_column = []
    for box, label in zip(boxes, clustering.labels_):
        if label < 0:
            outliers_column.append(box)
        else:
            boxes_column_ok.append(box)

    return boxes_column_ok, outliers_column


def detect_columns(options, file_image, file_boxes, file_columns, file_view):

    # parameters

    threshold_boxsize = options.threshold_boxsize # 0.3; width/height cannot deviate more than X from the median width/height
    threshold_x_deviation = options.threshold_x_deviation # 0.1; allowed x deviation (relative to median width) to form a column
    threshold_y_deviation = options.threshold_y_deviation # 2; allowed y deviation (relative to median height) to form a column
    threshold_min_samples = options.threshold_min_samples # 3; minimum characters to form a column
    detection_size = options.detection_size

    # read and rescale boxes if necessary

    view_image = Image.open(file_image).convert("RGBA")
    if detection_size > 0:
        width, height = view_image.size
        max_size = max(width, height)
        factor = 1
        if max_size != detection_size:
            factor = max_size / detection_size
        all_boxes = read_boxes(file_boxes, factor)
    else:
        all_boxes = read_boxes(file_boxes)

    # find median box

    sorted_boxes = sorted(all_boxes, key=lambda box: box.width * box.height)
    median_box = None
    if len(all_boxes) > 0:
        median_box = sorted_boxes[int(len(sorted_boxes) / 2)]

    # remove boxes that are too small or too large

    boxes_size_ok = []
    outliers_size = []
    for box in all_boxes:
        diff_width = abs(box.width - median_box.width) / median_box.width
        diff_height = abs(box.height - median_box.height) / median_box.height
        if diff_width <= threshold_boxsize or diff_height <= threshold_boxsize:
            boxes_size_ok.append(box)
        else:
            outliers_size.append(box)

    # remove boxes that are not part of a column

    boxes_column_ok = []
    outliers_column = []
    if len(boxes_size_ok) > 0:
        boxes_column_ok, outliers_column = cluster_columns(boxes_size_ok, median_box, threshold_x_deviation, threshold_y_deviation, threshold_min_samples)

    # save columns

    with open(file_columns, "w") as fd_tsv:
        writer_tsv = csv.writer(fd_tsv, delimiter="\t")
        for box in boxes_column_ok:
            writer_tsv.writerow(
                [
                    box.className,
                    box.xStart,
                    box.yStart,
                    box.xEnd,
                    box.yEnd,
                    box.className + ".png",
                    box.className,
                    ]
            )

    # create view

    draw = ImageDraw.Draw(view_image)
    for box in outliers_size:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="red")
    for box in outliers_column:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="blue")
    for box in boxes_column_ok:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="green")
    view_image.save(file_view, "PNG")


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
        help="Directory to find detection results",
    )
    parser.add_argument(
        "--dir-columns",
        dest="dir_columns",
        required=True,
        type=str,
        help="Directory to store column results",
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
    parser.add_argument(
        "--boxsize",
        dest="threshold_boxsize",
        required=False,
        type=float,
        default=0.3,
        help="Maximum deviation from the median box size",
    )
    parser.add_argument(
        "--x-deviation",
        dest="threshold_x_deviation",
        required=False,
        type=float,
        default=0.1,
        help="Maximum horizontal deviation in text column (relative to median box width)",
    )
    parser.add_argument(
        "--y-deviation",
        dest="threshold_y_deviation",
        required=False,
        type=float,
        default=2.0,
        help="Maximum vertical gap in text column (relative to median box height)",
    )
    parser.add_argument(
        "--min-samples",
        dest="threshold_min_samples",
        required=False,
        type=int,
        default=3,
        help="Minimum number of characters in text column",
    )

    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    img_dir = options.dir_images
    img_type_lower = options.image_type.lower()
    img_type_upper = options.image_type.upper()
    dir_detection = options.dir_detection
    dir_columns = options.dir_columns
    dir_columns_view = path.join(dir_columns, "view")

    if not path.exists(dir_columns):
        os.makedirs(dir_columns)
    if not path.exists(dir_columns_view):
        os.makedirs(dir_columns_view)

    print("Begin column detection..")
    files = sorted(os.listdir(img_dir))
    log_page = tqdm.tqdm(total=len(files), desc='Detect columns', position=1)
    for filename in files:
        if filename.endswith(".{}".format(img_type_lower)) or filename.endswith(".{}".format(img_type_upper)):
            file_image = path.join(img_dir, filename)
            file_boxes = path.join(dir_detection, "{}.tsv".format(filename.split(".")[0]))
            file_columns = path.join(dir_columns, "{}.tsv".format(filename.split(".")[0]))
            file_view = path.join(dir_columns_view, "{}.png".format(filename.split(".")[0]))
            if not path.exists(file_view):
                detect_columns(options, file_image, file_boxes, file_columns, file_view)
        log_page.update(1)

    print(".. done.")
