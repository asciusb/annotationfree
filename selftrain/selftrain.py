# Main method for annotation-free self-calibration.

import argparse
import os
import os.path as path
from shutil import copyfile
import lavd
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # required

    parser.add_argument(
        "--experiment",
        dest="experiment",
        required=True,
        type=str,
        help="Name of the experiment",
    )
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
        "--self-iter",
        dest="self_iter",
        required=True,
        type=int,
        help="Number of self iterations",
    )

    # optional (general)

    parser.add_argument(
        "--dir-root",
        dest="dir_root",
        required=False,
        type=str,
        default="../../experiments/",
        help="Experiments root directory",
    )
    parser.add_argument(
        "--font-name",
        dest="font_name",
        required=False,
        type=str,
        default="NomNaTongLight.ttf",
        help="Name of the font",
    )

    # optional (train)

    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        required=False,
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        required=False,
        type=int,
        default=1024,
        help="Size to which the longer image side is resized to",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        required=False,
        type=int,
        default=2,
        help="Number of pages per batch",
    )

    # optional (validation)

    parser.add_argument(
        "--dir-images-eval",
        dest="dir_images_eval",
        required=False,
        type=str,
        default="../../data/steles_evaluation/",
        help="Evaluation images root directory",
    )

    # optional (background)

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

    # optional (yolo)

    parser.add_argument(
        "--yolo-model",
        dest="yolo_model",
        required=False,
        type=str,
        default="yolov5m.pt",
        help="Yolo model (yolov5s.pt, yolov5m.pt, ...)",
    )
    parser.add_argument(
        "--yolo-confidence-eval",
        dest="yolo_confidence_eval",
        required=False,
        type=float,
        default=0.25,
        help="Yolo confidence threshold used for evaluation",
    )
    parser.add_argument(
        "--yolo-confidence-bg",
        dest="yolo_confidence_bg",
        required=False,
        type=float,
        default=0.25,
        help="Yolo confidence threshold used for background creation",
    )

    return parser


def train_yolo(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    batch_size = options.batch_size
    num_epochs = options.num_epochs
    image_size = options.image_size
    total_epochs = (iteration + 1) * num_epochs
    yolo_model = options.yolo_model

    dir_experiment = path.join(dir_root, experiment)
    file_train = path.join(dir_experiment, "yolo", "gt.yaml")
    file_hyp = "../yolov5/data/hyp.finetune.yaml"
    file_model = path.join(dir_experiment, "model_epoch_{}.pt".format(total_epochs))

    if not path.exists(file_train):
        args = "--experiment {}".format(dir_experiment)
        cmd = "python vhist2yolo.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(file_train):
            raise ValueError("Could not find yolo train file: {}".format(file_train))

    if not path.exists(file_model):
        file_model_log = "runs/train/{}/weights/last.pt".format(experiment)
        if path.exists(file_model_log):
            copyfile(file_model_log, file_model)
        else:
            args = "--epochs {}".format(num_epochs) # 15
            args += " --data {}".format(file_train)
            args += " --img-size {}".format(image_size) # 1024
            args += " --batch {}".format(batch_size) # 24
            args += " --name {}".format(experiment)
            args += " --rect"
            args += " --single-cls"
            args += " --hyp {}".format(file_hyp) # data/hyp.finetune.yaml
            # start training
            if iteration == 0:
                args += " --weights {}".format(yolo_model) # yolov5m.pt
            # continue training
            else:
                previous_experiment = "{}_iter_{:02d}".format(options.experiment, iteration - 1)
                previous_epochs = total_epochs - num_epochs
                previous_model = path.join(dir_root, previous_experiment, "model_epoch_{}.pt".format(previous_epochs))
                args += " --weights {}".format(previous_model)
            cmd = "conda run -n yolo python ../yolov5/train.py {}".format(args)
            print(cmd)
            os.system(cmd)

        if path.exists(file_model_log):
            copyfile(file_model_log, file_model)
        else:
            raise ValueError("Could not find model: {}".format(file_model_log))


def validate_yolo(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    dir_images_eval = options.dir_images_eval
    image_type = options.image_type
    image_size = options.image_size
    num_epochs = options.num_epochs
    total_epochs = (iteration + 1) * num_epochs
    yolo_confidence_eval = options.yolo_confidence_eval

    dir_experiment = path.join(dir_root, experiment)
    dir_eval = path.join(dir_experiment, "eval_epoch_{}".format(total_epochs))
    dir_view = path.join(dir_eval, "view")
    file_model = path.join(dir_experiment, "model_epoch_{}.pt".format(total_epochs))

    dir_yolo_detected = "runs/detect/{}_eval/".format(experiment)
    if not path.exists(dir_yolo_detected):
        args = " --weights {}".format(file_model)
        args += " --source {}".format(dir_images_eval)
        args += " --img-size {}".format(image_size)
        args += " --name {}_eval".format(experiment)
        args += " --save-txt"
        args += " --save-conf"
        args += " --conf-thres {}".format(yolo_confidence_eval)
        cmd = "conda run -n yolo python ../yolov5/detect.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_yolo_detected):
            raise ValueError("Could not find yolo validation results: {}".format(dir_yolo_detected))

    if not path.exists(dir_view):
        args = " --dir-images {}".format(dir_images_eval)
        args += " --image-type {}".format(image_type)
        args += " --image-size {}".format(image_size)
        args += " --yolo-detected {}".format(dir_yolo_detected)
        args += " --vhist-detected {}".format(dir_eval)
        args += " --vhist-view {}".format(dir_view)
        cmd = "python yolo2vhist.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_view):
            raise ValueError("Could not find yolo validation view: {}".format(dir_view))


def layout(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    dir_images_eval = options.dir_images_eval
    detection_method = options.detection_method
    image_type = options.image_type
    image_size = options.image_size
    num_epochs = options.num_epochs
    total_epochs = (iteration + 1) * num_epochs

    dir_experiment = path.join(dir_root, experiment)
    dir_eval = path.join(dir_experiment, "eval_epoch_{}".format(total_epochs))
    dir_columns = path.join(dir_eval, "columns")

    if not path.exists(dir_columns):
        args = " --dir-images {}".format(dir_images_eval)
        args += " --image-type {}".format(image_type)
        args += " --dir-detection {}".format(dir_eval)
        args += " --dir-columns {}".format(dir_columns)
        if detection_method == "fcos":
            args += " --detection-size {}".format(image_size)
        cmd = "python column.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_columns):
            raise ValueError("Could not find layout analysis results: {}".format(dir_columns))


def evaluate(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    dir_images_eval = options.dir_images_eval
    image_type = options.image_type
    font_name = options.font_name
    num_epochs = options.num_epochs
    total_epochs = (iteration + 1) * num_epochs

    dir_experiment = path.join(dir_root, experiment)
    dir_eval = path.join(dir_experiment, "eval_epoch_{}".format(total_epochs))
    dir_columns = path.join(dir_eval, "columns")
    dir_evaluation = path.join(dir_eval, "results")
    file_font = path.join(dir_root, "fonts", font_name)

    if not path.exists(dir_evaluation):
        args = " --dir-images {}".format(dir_images_eval)
        args += " --image-type {}".format(image_type)
        args += " --dir-detection {}".format(dir_columns)
        args += " --dir-evaluation {}".format(dir_evaluation)
        args += " --font {}".format(file_font)
        cmd = "python evaluate.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_evaluation):
            raise ValueError("Could not find evaluation results: {}".format(dir_evaluation))


def detect_yolo(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    dir_images = options.dir_images
    image_type = options.image_type
    image_size = options.image_size
    num_epochs = options.num_epochs
    total_epochs = (iteration + 1) * num_epochs
    yolo_confidence_bg = options.yolo_confidence_bg

    dir_experiment = path.join(dir_root, experiment)
    dir_detected = path.join(dir_experiment, "detected")
    dir_view = path.join(dir_experiment, "detected_view")
    file_model = path.join(dir_experiment, "model_epoch_{}.pt".format(total_epochs))

    dir_yolo_detected = "runs/detect/{}_detected/".format(experiment)
    if not path.exists(dir_yolo_detected):
        args = " --weights {}".format(file_model)
        args += " --source {}".format(dir_images)
        args += " --img-size {}".format(image_size)
        args += " --name {}_detected".format(experiment)
        args += " --save-txt"
        args += " --save-conf"
        args += " --conf-thres {}".format(yolo_confidence_bg)
        cmd = "conda run -n yolo python ../yolov5/detect.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_yolo_detected):
            raise ValueError("Could not find yolo detection results: {}".format(dir_yolo_detected))

    if not path.exists(dir_view):
        args = " --dir-images {}".format(dir_images)
        args += " --image-type {}".format(image_type)
        args += " --image-size {}".format(image_size)
        args += " --yolo-detected {}".format(dir_yolo_detected)
        args += " --vhist-detected {}".format(dir_detected)
        args += " --vhist-view {}".format(dir_view)
        cmd = "python yolo2vhist.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_view):
            raise ValueError("Could not find yolo detection view: {}".format(dir_view))


def background(options, iteration):
    experiment = "{}_iter_{:02d}".format(options.experiment, iteration)

    dir_root = options.dir_root
    image_type = options.image_type
    threshold_boxsize = options.threshold_boxsize
    threshold_x_deviation = options.threshold_x_deviation
    threshold_y_deviation = options.threshold_y_deviation
    threshold_min_samples = options.threshold_min_samples
    threshold_min_boxes = options.threshold_min_boxes

    dir_experiment = path.join(dir_root, experiment)
    dir_background = path.join(dir_experiment, "background_images")

    if not path.exists(dir_background):
        args = "--experiment {}".format(dir_experiment)
        args += " --image-type {}".format(image_type)
        args += " --boxsize {}".format(threshold_boxsize)
        args += " --x-deviation {}".format(threshold_x_deviation)
        args += " --y-deviation {}".format(threshold_y_deviation)
        args += " --min-samples {}".format(threshold_min_samples)
        args += " --min-boxes {}".format(threshold_min_boxes)
        cmd = "python background.py {}".format(args)
        print(cmd)
        os.system(cmd)

        if not path.exists(dir_background):
            raise ValueError("Could not find background results: {}".format(dir_background))


if __name__ == "__main__":
    options = build_parser().parse_args()
    self_iter = options.self_iter
    num_epochs = options.num_epochs
    experiment = options.experiment
    

    for iteration in range(self_iter):
        logstep = iteration + 1
        total_epochs = (iteration + 1) * num_epochs

        experiment = "{}_iter_{:02d}".format(options.experiment, iteration)
        dir_experiment_iter = path.join(options.dir_root, experiment)
        print("\n*** Self-training {}  ***\n".format(experiment))

        print("{}: train ..".format(experiment))
        train_yolo(options, iteration)
        print(".. done.")

        dir_eval = path.join(dir_experiment_iter, "eval_epoch_{}".format(total_epochs))

        print("{}: validate ..".format(experiment))
        validate_yolo(options, iteration)
        print(".. done.")

        print("{}: analyze column layout ..".format(experiment))
        layout(options, iteration)
        print(".. done.")

        print("{}: evaluate performance ..".format(experiment))
        evaluate(options, iteration)
        print(".. done.")

        print("{}: detect whole dataset ..".format(experiment))
        detect_yolo(options, iteration)
        print(".. done.")

        print("{}: create background pages ..".format(experiment))
        background(options, iteration)
        print(".. done.")

