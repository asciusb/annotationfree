# Convert vhist format to yolo.

import csv
import os
import os.path as path
from shutil import copyfile
from PIL import Image
import argparse
import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # required

    parser.add_argument(
        "--experiment",
        dest="experiment",
        required=True,
        type=str,
        help="Experiment directory",
    )

    return parser


if __name__ == "__main__":
    options = build_parser().parse_args()
    experiment = options.experiment
    print("\n*** Transform VHIST to YOLO: {} ***\n".format(experiment))
    
    # vhist files
    d_vhist = {"train": path.join(experiment, "train"), "val": path.join(experiment, "val")}
    f_vhist_gt = {"train": path.join(d_vhist["train"], "gt_nclass.tsv"), "val": path.join(d_vhist["val"], "gt_nclass.tsv")}
    f_vhist_classes = path.join(d_vhist["train"], "classes_nclass.tsv")
    
    # yolo files
    d_yolo = path.join(experiment, "yolo")
    d_yolo_images = {"train": path.join(d_yolo, "images", "train"), "val": path.join(d_yolo, "images", "val")}
    d_yolo_labels = {"train": path.join(d_yolo, "labels", "train"), "val": path.join(d_yolo, "labels", "val")}
    f_yolo_gt = path.join(d_yolo, "gt.yaml")
    if not path.exists(d_yolo):
        os.makedirs(d_yolo)
    
    # classes
    classes = []
    char2id = {}
    with open(f_vhist_classes, "r") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        for i, line in enumerate(reader):
            char = line[0]
            classes.append(char)
            char2id[char] = i
            
    # yolo gt
    with open(f_yolo_gt, "w") as f:
        f.write("train: {}\n".format(d_yolo_images["train"]))
        f.write("val: {}\n".format(d_yolo_images["val"]))
        f.write("nc: {}\n".format(len(classes)))
        names = []
        for char in classes:
            names.append("'{}'".format(char))
        f.write("names: [ {} ]\n".format(", ".join(names)))
        
    pbar_pos = 0
    for dataset in ["train", "val"]:
        pbar_pos += 1
        if not path.exists(d_yolo_images[dataset]):
            os.makedirs(d_yolo_images[dataset])
        if not path.exists(d_yolo_labels[dataset]):
            os.makedirs(d_yolo_labels[dataset])
    
        # vhist gt
        vhist_gt = []
        with open(f_vhist_gt[dataset], "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
            for line in reader:
                vhist_gt.append(line)
            
        pbar = tqdm.tqdm(total=len(vhist_gt), desc="create yolo {}".format(dataset), position=pbar_pos)
        for sample in vhist_gt:
            f_vhist_image = path.join(d_vhist[dataset], sample[0])
            f_vhist_label = path.join(d_vhist[dataset], sample[1])

            f_yolo_image = path.join(d_yolo_images[dataset], sample[0])
            f_yolo_label = path.join(d_yolo_labels[dataset], "{}.txt".format(sample[1].split("_nclass.tsv")[0]))
            
            # yolo image
            im = Image.open(f_vhist_image)
            im_width, im_height = im.size
            copyfile(f_vhist_image, f_yolo_image)
            
            # yolo labels
            labels = []
            with open(f_vhist_label, "r") as f:
                reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
                for line in reader:
                    char = line[0]
                    x1 = int(line[1])
                    y1 = int(line[2])
                    x2 = int(line[3])
                    y2 = int(line[4])
                    if char in char2id:
                        char_id = char2id[char]
                    else:
                        char_id = 0 # allow unknown classes
                    x_center = ((x1 + x2) / 2.0) / im_width
                    y_center = ((y1 + y2) / 2.0) / im_height
                    width = (x2 - x1) / im_width
                    height = (y2 - y1) / im_height
                    label = [char_id, x_center, y_center, width, height]
                    if label not in labels: # do not allow duplicates
                        labels.append(label)
            with open(f_yolo_label, "w") as f:
                writer_tsv = csv.writer(f, delimiter=" ")
                sorted_labels = sorted(labels, key=lambda label: label[0])
                for label in sorted_labels:
                    writer_tsv.writerow(label)
                    
            pbar.update(1)
