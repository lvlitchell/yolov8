import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from random import randint
import matplotlib as mpl
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
mpl.use('TkAgg')


def rgbd_to_depth(rgb_depth_image):
    """converts a bgr cv2 color depth maps into a black and white
        depth map.
        inputs: -depth: numpy array shape (H, W, 3)
        outputs: -depth_bw: numpy array shape (H, W)
    """
    b, g, r = np.transpose(rgb_depth_image, (2, 0, 1))
    depth_bw = np.bitwise_or(np.left_shift(g, 8), b)
    depth_bw = np.asarray(depth_bw, np.float32)

    return depth_bw


def mask_to_polygons(mask):
    # Calcula los contornos
    mask = mask.astype(bool)
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # convertimos los contornos a polígonos de Label Studio
    polygons = []
    normalized_polygons = []
    for contour in contours:
        # Lo meto en un try porque la extraccion de polígonos que hace el opencv a partir de la máscara
        # a veces genera polígonos de menos de 4 vértices, que no tiene sentido por no ser cerrados,
        # provocando que falle al convertir a polígno de shapely
        try:
            polygon = contour.reshape(-1, 2).tolist()
            # normalizamos las coordenadas entre 0 y 1 porque así lo requiere YOLOv8
            normalized_polygon = [[round(coord[0] / mask.shape[1], 4), round(coord[1] / mask.shape[0], 4)] for coord in
                                  polygon]
            # Convertimos a objeto poligono de shapely (sin normalizar)
            polygon_shapely = Polygon(polygon)
            simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
            polygons.append(list(simplified_polygon.exterior.coords))
            # normalizdos
            normalized_polygons.append(list(Polygon(normalized_polygon).exterior.coords))
        except (Exception, ):
            pass

    return polygons, normalized_polygons


def plot_polygon_on_image(image, polygon):
    polygon = np.asarray(polygon)
    polygon = np.squeeze(polygon)
    image = cv2.polylines(image, [polygon.astype('int')], True, (randint(0, 255), randint(0, 255), randint(0, 255)),
                  2)  # Draw Poly Lines
    image = cv2.fillPoly(image, [polygon.astype('int')], (randint(0,255),randint(0,255),randint(0,255)), cv2.LINE_AA) # Draw area

    return image


def convert_image_to_annotation(image, visualize=False):
    """Converts the red hand images of nyu to a mask and then a yolo formatted annotation
       inputs: -image numpy array red hand image
       outputs:
    """
    r = image[..., -1]
    mask = np.zeros_like(r)
    mask[r == 255] = 1
    poly, normalized_poly = mask_to_polygons(mask)
    if visualize:
        image = plot_polygon_on_image(image, poly)
        plt.imshow(image)
        plt.show()

    return normalized_poly


def create_yolo_annotations(image_dir, ann_dir, class_id=0):
    image_dir = Path(image_dir)
    ann_dir = Path(ann_dir)
    assert image_dir.is_dir()
    assert ann_dir.is_dir()

    for image_file in image_dir.iterdir():
        try:
            image = cv2.imread(image_file.as_posix())
        except (Exception, ):
            print(f"unable to load {image_file.name}, skipping!")
            continue
        polygon = convert_image_to_annotation(image)
        if len(polygon) > 1:
            polygon = polygon[0]
        polygon = np.asarray(polygon).flatten().tolist()
        # print(polygon)
        txt_file = (ann_dir / image_file.with_suffix('.txt').name).as_posix()
        with open(txt_file, 'w') as file:
            # Convert the list elements to strings and join them with spaces
            line = f'{str(class_id)} ' + ' '.join(map(str, polygon))
            file.write(line)

        # plt.imshow(image)
        # plt.show()


def create_yolo_overview_txt(image_dir, dataset_dir):
    image_dir = Path(image_dir)
    dataset_dir = Path(dataset_dir)
    assert image_dir.is_dir()
    assert dataset_dir.is_dir()
    txt_file_path = (dataset_dir / image_dir.name).with_suffix('.txt').as_posix()
    txt_file = open(txt_file_path, "w")

    for image_file in image_dir.iterdir():
        txt_file.writelines(f"{image_file.as_posix()} \n")

    txt_file.close()


def test_yolo_annotation(image_dir, ann_dir):
    image_dir = Path(image_dir)
    ann_dir = Path(ann_dir)
    assert image_dir.is_dir()
    assert ann_dir.is_dir()

    for image_file in image_dir.iterdir():
        try:
            image = cv2.imread(image_file.as_posix())
            h, w = image.shape[0:2]
        except (Exception, ):
            print(f"unable to load {image_file.name}, skipping!")
            continue
        txt_file = (ann_dir / image_file.with_suffix('.txt').name).as_posix()
        print("txt_file name", txt_file)
        with open(txt_file, "r") as f:
            labels = f.read().splitlines()

        for label in labels:
            class_id, *poly = label.split(' ')
            print("len of poly", len(poly))
            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)  # Read poly, reshape
            print("number of pts in seg", poly.shape)
            poly *= [w, h]  # Unscale
            cv2.polylines(image, [poly.astype('int')], True, (randint(0, 255), randint(0, 255), randint(0, 255)),
                          2)  # Draw Poly Lines
            cv2.fillPoly(image, [poly.astype('int')], (randint(0, 255), randint(0, 255), randint(0, 255)), cv2.LINE_AA)
        plt.imshow(image)
        plt.show()


def convert_images_to_bw(image_dir, visualize=False):
    image_dir = Path(image_dir)
    for image_file in image_dir.iterdir():
        try:
            image = cv2.imread(image_file.as_posix())
        except (Exception, ):
            print(f"unable to load image {image_file.name}, skipping!")
        bw_image = image[:, :, 0].copy()
        if visualize:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(image)
            ax[1].imshow(bw_image)
            plt.show()
        cv2.imwrite(image_file.as_posix(), bw_image)


def restructure_yolo_data(split="test"):
    bw_image_dir = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand_bw/{split}/images/"
    bw_label_dir = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand_bw/{split}/labels/"
    rgb_images_dir = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand/images/{split}"
    label_txt = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand_bw/{split}.txt"
    with open(label_txt, "r") as file:
        for path in file:
            path = path.strip()
            image_file = Path(rgb_images_dir) / Path(path).name
            new_path = Path(bw_image_dir) / Path(path).name
            image = cv2.imread(image_file.as_posix())
            bw_image = image[:, :, 0].copy()
            cv2.imwrite(filename=new_path.as_posix(), img=bw_image)
    create_yolo_annotations(image_dir=rgb_images_dir, ann_dir=bw_label_dir)



if __name__ == "__main__":
    split = "test"
    # dataset_dir = "/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand_bw/"
    image_dir = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand_bw/{split}/images"
    ann_dir = f"/home/inseer/data/Hand_Testing/yolov8_seg/nyu_hand/annotations/{split}"
    # create_yolo_overview_txt(image_dir, dataset_dir)
    # restructure_yolo_data(split)
    test_yolo_annotation(image_dir, ann_dir)

