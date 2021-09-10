# Cluster detected characters to columns and create background images.

import argparse
import csv
import os
import os.path as path
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import measurements
import tqdm


def read_boxes(path_boxes):
    with open(path_boxes, "r") as fd:
        reader = csv.reader(
            fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
        )
        boxes = []
        for bbox in reader:
            box = Box(bbox)
            boxes.append(box)
    return boxes


class Cluster:

    def __init__(self):
        self.boxes = []

    def append(self, box):
        self.boxes.append(box)

    def sort(self):
        self.boxes = sorted(self.boxes, key=lambda box: box.yCenter)

    def size(self):
        return len(self.boxes)

    def xMean(self):
        x = 0
        for box in self.boxes:
            x += box.xCenter
        x /= self.size()
        return x

    def __str__(self):
        str = "Cluster "
        if len(self.boxes) > 0:
            str += "xCenter0={:.0f}, yCenter0={:.0f} ".format(self.boxes[0].xCenter, self.boxes[0].yCenter)
        str += "({} boxes)".format(self.size())
        return str


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
        self.area = self.width * self.height

    def union(self, other):
        uclassName = self.className
        uxStart = min(self.xStart, other.xStart)
        uyStart = min(self.yStart, other.yStart)
        uxEnd = max(self.xEnd, other.xEnd)
        uyEnd = max(self.yEnd, other.yEnd)
        return Box([uclassName, uxStart, uyStart, uxEnd, uyEnd])

    def __str__(self):
        return "{}_{}_{}_{}_{}_area={}".format(self.className, self.xStart, self.yStart, self.xEnd, self.yEnd, self.area)


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

    clusters = {}
    boxes_column_ok = []
    outliers_column = []
    for box, label in zip(boxes, clustering.labels_):
        if label < 0:
            outliers_column.append(box)
        else:
            if not label in clusters:
                clusters[label] = Cluster()
            clusters[label].append(box)
            boxes_column_ok.append(box)

    columns = []
    for label in clusters:
        column = None
        for box in clusters[label].boxes:
            if column is None:
                column = box
            else:
                column = column.union(box)
        columns.append(column)

    max_chars_per_column = 0
    for label in clusters:
        numchars = clusters[label].size()
        if numchars > max_chars_per_column:
            max_chars_per_column = numchars

    return columns, boxes_column_ok, outliers_column, max_chars_per_column


def get_empty_patch(columns, text_area, source_img):
    beam_width = 5
    min_width = 10
    min_height = 20
    min_color = 128
    patch_candidates = []
    for i, column in enumerate(columns):

        # below column

        label = "below_{}_{}".format(i, 0)
        x1 = column.xStart
        y1 = column.yEnd
        x2 = column.xEnd
        y2 = text_area.yEnd
        patch_candidates.append(Box([label, x1, y1, x2, y2]))
        jmax = min(i + beam_width, len(columns))
        for j in range (i+1, jmax):
            column2 = columns[j]
            label = "below_{}_{}".format(i, j)
            y1 = max(y1, column2.yEnd)
            x2 = column2.xEnd
            box = Box([label, x1, y1, x2, y2])
            patch_candidates.append(box)

        # above column

        label = "above_{}_{}".format(i, 0)
        x1 = column.xStart
        y1 = text_area.yStart
        x2 = column.xEnd
        y2 = column.yStart
        patch_candidates.append(Box([label, x1, y1, x2, y2]))
        jmax = min(i + beam_width, len(columns))
        for j in range (i+1, jmax):
            column2 = columns[j]
            label = "above_{}_{}".format(i, j)
            y1 = min(y1, column2.yStart)
            x2 = column2.xEnd
            box = Box([label, x1, y1, x2, y2])
            patch_candidates.append(box)

        # between columns

        if i + 1 < len(columns):
            label = "between_{}_{}".format(i, i+1)
            column2 = columns[i+1]
            x1 = column.xEnd
            y1 = text_area.yStart
            x2 = column2.xStart
            y2 = text_area.yEnd
            patch_candidates.append(Box([label, x1, y1, x2, y2]))

    if len(patch_candidates) == 0:
        return None

    result = None
    result_value = None
    for cand in patch_candidates:
        size_ok = cand.width > min_width and cand.height > min_height
        if size_ok:
            region = source_img.crop((cand.xStart, cand.yStart, cand.xEnd, cand.yEnd))
            npregion = np.array(region.convert('L'))
            mean = measurements.mean(npregion)
            stdv = measurements.standard_deviation(npregion)
            val = stdv
            color_ok = mean > min_color
            if (result is None or val < result_value) and color_ok:
                result = cand
                result_value = val

    return result


def clear_text_area(source_img, text_area, empty_patch):
    patch = source_img.crop((empty_patch.xStart, empty_patch.yStart, empty_patch.xEnd, empty_patch.yEnd))
    stepx = text_area.width // empty_patch.width
    stepy = text_area.height // empty_patch.height
    for i in range(0, stepx + 1):
        for j in range(0, stepy + 1):
            x1 = text_area.xStart + (i * empty_patch.width)
            y1 = text_area.yStart + (j * empty_patch.height)
            if i == stepx:
                x1 = text_area.xEnd - empty_patch.width
            if j == stepy:
                y1 = text_area.yEnd - empty_patch.height
            x2 = x1 + empty_patch.width
            y2 = y1 + empty_patch.height
            source_img.paste(patch, (x1, y1, x2, y2))


def detect_background(options, file_image, file_boxes, file_background, file_chars, file_area, file_view):

    # parameters

    threshold_boxsize = options.threshold_boxsize # 0.3; width/height cannot deviate more than X from the median width/height
    threshold_x_deviation = options.threshold_x_deviation # 0.1; allowed x deviation (relative to median width) to form a column
    threshold_y_deviation = options.threshold_y_deviation # 2; allowed y deviation (relative to median height) to form a column
    threshold_min_samples = options.threshold_min_samples # 3; minimum characters to form a column
    threshold_min_boxes = options.threshold_min_boxes # 100; minimum boxes per page

    # find median box

    all_boxes = read_boxes(file_boxes)
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

    # create columns

    columns = []
    boxes_column_ok = []
    outliers_column = []
    max_chars_per_column = 0
    if len(boxes_size_ok) > 0:
        columns, boxes_column_ok, outliers_column, max_chars_per_column = cluster_columns(boxes_size_ok, median_box, threshold_x_deviation, threshold_y_deviation, threshold_min_samples)
        if len(columns) > 0:
            columns, _, _, _ = cluster_columns(columns, median_box, 3 * threshold_x_deviation, 100, 1)
            columns = sorted(columns, key=lambda box: box.xCenter)

    # create text area

    text_area = None
    total_chars = len(boxes_column_ok)
    if total_chars > threshold_min_boxes:
        for column in columns:
            if text_area is None:
                text_area = column
            else:
                text_area = text_area.union(column)

    # create background

    background_image = Image.open(file_image).convert("RGBA")
    width, height = background_image.size
    empty_patch = None
    if len(columns) > 0 and text_area is not None:
        empty_patch = get_empty_patch(columns, text_area, background_image)
        if empty_patch is not None:
            clear_text_area(background_image, text_area, empty_patch)

            factor = 1
            max_size = max(width, height)
            if max_size > 1024:
                factor = 1024 / max_size
                background_image.thumbnail((1024, 1024))

            background_image.save(file_background, "PNG")
            with open(file_chars, "w") as fd_tsv:
                writer_tsv = csv.writer(fd_tsv, delimiter="\t")
                name = file_background.split("/")[-1]
                writer_tsv.writerow([name, len(columns), max_chars_per_column, len(all_boxes), len(boxes_size_ok), len(boxes_column_ok), median_box.width, median_box.height, factor])
            with open(file_area, "w") as fd_tsv:
                writer_tsv = csv.writer(fd_tsv, delimiter="\t")
                writer_tsv.writerow([int(text_area.xStart * factor), int(text_area.yStart * factor), int(text_area.xEnd * factor), int(text_area.yEnd * factor)])

    # create view

    view_image = Image.open(file_image).convert("RGBA")
    draw = ImageDraw.Draw(view_image)
    for box in outliers_size:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="red")
    for box in outliers_column:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="blue")
    for box in boxes_column_ok:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=3, fill=None, outline="green")
    for box in columns:
        draw.rectangle(((box.xStart, box.yStart), (box.xEnd, box.yEnd)), width=4, fill=None, outline="yellow")
    if text_area is not None:
        draw.rectangle(((text_area.xStart, text_area.yStart), (text_area.xEnd, text_area.yEnd)), width=4, fill=None, outline="yellow")
    if empty_patch is not None:
        draw.rectangle(((empty_patch.xStart, empty_patch.yStart), (empty_patch.xEnd, empty_patch.yEnd)), width=8, fill=None, outline="cyan")
    view_image.save(file_view, "PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # required

    parser.add_argument(
        "--experiment",
        dest="exp_dir",
        required=True,
        type=str,
        help="Experiment directory",
    )
    parser.add_argument(
        "--image-type",
        dest="img_type",
        required=True,
        type=str,
        help="Image type (jpg, png, ...)",
    )

    # optional

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
    parser.add_argument(
        "--min-boxes",
        dest="threshold_min_boxes",
        required=False,
        type=int,
        default=100,
        help="Minimum number of selected boxes on page",
    )

    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    exp_dir = options.exp_dir
    img_type_lower = options.img_type.lower()
    img_type_upper = options.img_type.upper()

    dir_detected = path.join(exp_dir, "detected")
    dir_background = path.join(exp_dir, "background_images")
    dir_background_chars = path.join(exp_dir, "background_chars")
    dir_background_area = path.join(exp_dir, "background_areas")
    dir_background_view = path.join(exp_dir, "background_view")
    if not path.exists(dir_background):
        os.makedirs(dir_background)
    if not path.exists(dir_background_chars):
        os.makedirs(dir_background_chars)
    if not path.exists(dir_background_area):
        os.makedirs(dir_background_area)
    if not path.exists(dir_background_view):
        os.makedirs(dir_background_view)

    print("Detect background ..")
    files = sorted(os.listdir(dir_detected))
    log_page = tqdm.tqdm(total=len(files), desc='Create background', position=1)
    for filename in files:
        if filename.endswith(".{}".format(img_type_lower)) or filename.endswith(".{}".format(img_type_upper)):
            file_image = os.path.join(dir_detected, filename)
            file_boxes = os.path.join(dir_detected, "{}.tsv".format(filename.split(".")[0]))
            file_background = os.path.join(dir_background, "{}.png".format(filename.split(".")[0]))
            file_chars = os.path.join(dir_background_chars, "{}.tsv".format(filename.split(".")[0]))
            file_area = os.path.join(dir_background_area, "{}.tsv".format(filename.split(".")[0]))
            file_view = os.path.join(dir_background_view, "{}.png".format(filename.split(".")[0]))
            detect_background(options, file_image, file_boxes, file_background, file_chars, file_area, file_view)
        log_page.update(1)

    file_colchars = os.path.join(dir_background, "background_col_char.tsv")
    colchars = []
    for filename in sorted(os.listdir(dir_background_chars)):
        if filename.endswith(".tsv"):
            file_tsv = os.path.join(dir_background_chars, filename)
            with open(file_tsv, "r") as fd:
                reader = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
                for line in reader:
                    colchars.append([line[0], line[1], line[2]])
    with open(file_colchars, 'w') as fd_tsv:
        writer_tsv = csv.writer(fd_tsv, delimiter="\t")
        for line in colchars:
            writer_tsv.writerow(line)

