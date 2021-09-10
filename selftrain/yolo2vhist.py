# Convert yolo format to vhist.

import csv
import os
from shutil import copyfile
from PIL import Image
from PIL.Image import LANCZOS
import argparse
import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # required

    parser.add_argument(
        "--dir-images",
        dest="dir_images",
        required=True,
        type=str,
        help="Original images",
    )
    parser.add_argument(
        "--image-type",
        dest="image_type",
        required=True,
        type=str,
        help="Image type (jpg, png, ...)",
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        required=True,
        type=int,
        help="Size to which the longer image side is resized to",
    )
    parser.add_argument(
        "--yolo-detected",
        dest="yolo_detected",
        required=True,
        type=str,
        help="Yolo detection results",
    )
    parser.add_argument(
        "--vhist-detected",
        dest="vhist_detected",
        required=True,
        type=str,
        help="VHIST detection",
    )
    parser.add_argument(
        "--vhist-view",
        dest="vhist_view",
        required=True,
        type=str,
        help="VHIST view",
    )

    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    dir_images = options.dir_images
    img_type_lower = options.image_type.lower()
    img_type_upper = options.image_type.upper()
    image_size = options.image_size
    yolo_detected = options.yolo_detected
    vhist_detected = options.vhist_detected
    vhist_view = options.vhist_view
    if not os.path.exists(vhist_detected):
        os.makedirs(vhist_detected)
    if not os.path.exists(vhist_view):
        os.makedirs(vhist_view)

    print("\n*** Transform YOLO to VHIST: {} ***\n".format(yolo_detected))

    files = sorted(os.listdir(dir_images))
    log_page = tqdm.tqdm(total=len(files), desc='YOLO to VHIST', position=1)
    for filename in files:
        if filename.endswith(".{}".format(img_type_lower)) or filename.endswith(".{}".format(img_type_upper)):

            # view image
            f_yolo_view = os.path.join(yolo_detected, filename)
            f_vhist_view = os.path.join(vhist_view, filename)
            copyfile(f_yolo_view, f_vhist_view)

            # detected image
            f_original_img = os.path.join(dir_images, filename)
            f_vhist_img = os.path.join(vhist_detected, filename)
            copyfile(f_original_img, f_vhist_img)

            # detected tsv
            f_yolo_boxes = os.path.join(yolo_detected, "labels", "{}.txt".format(filename.split(".")[0]))
            f_vhist_tsv = os.path.join(vhist_detected, "{}.tsv".format(filename.split(".")[0]))
            original_img = Image.open(f_original_img)
            im_width, im_height = original_img.size
            labels = []
            if os.path.exists(f_yolo_boxes):
                with open(f_yolo_boxes, "r") as f:
                    reader = csv.reader(f, delimiter=" ", quoting=csv.QUOTE_NONE, quotechar="")
                    for line in reader:
                        char_id = line[0]
                        x_center = int(float(line[1]) * im_width)
                        y_center = int(float(line[2]) * im_height)
                        width = int(float(line[3]) * im_width)
                        height = int(float(line[4]) * im_height)
                        x1 = int(x_center - (width / 2))
                        y1 = int(y_center - (height / 2))
                        x2 = int(x_center + (width / 2))
                        y2 = int(y_center + (height / 2))
                        label = [char_id, x1, y1, x2, y2, "{}.png".format(char_id), char_id]
                        labels.append(label)
            with open(f_vhist_tsv, "w") as f:
                writer_tsv = csv.writer(f, delimiter="\t")
                for label in labels:
                    writer_tsv.writerow(label)

            log_page.update(2)
