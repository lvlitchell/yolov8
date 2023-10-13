import cv2
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ultralytics import YOLO
from NYU_Dataset_Conversion import plot_polygon_on_image
from types import SimpleNamespace
import time

mpl.use('TkAgg')


def train_model(data_yaml):
    # Load a model
    model = YOLO("yolov8m-seg.pt")  # load pretrained segmentation model
    # # train model
    model.train(data=data_yaml, epochs=100,
                project="EgoHands",
                name="Hand_Crops_imgsz_256",
                imgsz=256
                )  # train the model
    model.val()  # evaluate model performance on the validation set


def get_inference_args():
    args = SimpleNamespace()
    args.conf = 0.25
    args.iou = 0.7
    args.imgsz = 640
    args.half = False
    args.device = '0'
    args.show = True
    args.save = False
    args.save_txt = False
    args.max_det = 20
    args.boxes = False
    args.retina_masks = False
    args.visualize = False
    args.augment = False
    args.classes = None

    return args


def test_model(weights_path, image_dir, show=False):
    image_dir = Path(image_dir)
    save_dir = Path(image_dir).parent / "yolo_segs"
    save_dir.mkdir(exist_ok=True, parents=True)
    assert image_dir.is_dir()
    # Load a model
    model = YOLO(weights_path)  # load pretrained segmentation model
    # predict with model
    for image_path in image_dir.iterdir():
        image_ext = image_path.suffix
        print("image name", image_path.name)
        if image_ext == ".npy":
            image = np.load(image_path.as_posix(), allow_pickle=True)
        else:
            image = cv2.imread(image_path.as_posix())

        results = model.predict(source=image, conf=0.5, iou=0.4)
        seg = results[0]
        if seg:
            all_segs = []
            for i in range(len(seg.masks.xy)):
                seg_xy = seg.masks.xy[i]
                all_segs.append(seg_xy)
                image = plot_polygon_on_image(image, seg_xy)
            seg_save_path = save_dir / image_path.with_suffix('.npy').name
            np.save(seg_save_path.as_posix(), all_segs, allow_pickle=True)
            if show:
                plt.imshow(image)
                plt.show()
        else:
            print(f"no results for image {image_path.name}!")



if __name__ == "__main__":
    weights_path = "/home/inseer/engineering/yolov8/EgoHands/Hand_Crops_imgsz_256/weights/best.pt"
    image_dir = "/home/inseer/data/Hand_Testing/Orientation/Mitchell_Waving/hand_crops"
    test_model(weights_path, image_dir, show=True)
    # data_yaml = "/home/inseer/Downloads/ego_hands_crops_yolov8/ego_hands_crops.yaml"
    # train_model(data_yaml)




