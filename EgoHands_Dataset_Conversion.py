import cv2
import scipy.io as sio
import numpy as np
import shutil
from pathlib import Path
from NYU_Dataset_Conversion import test_yolo_annotation


def add_zeros(integer_string):
    integer_string = str(integer_string)
    num_zeros_to_prepend = 4 - len(integer_string)
    if num_zeros_to_prepend < 0:
        return integer_string
    result = '0' * num_zeros_to_prepend + integer_string
    return result


def convert_vid_label_2_dict(vid_label):
    vid_dict = {}
    vid_dict["video_id"] = vid_label[0]
    vid_dict["partner_video_id"] = vid_label[1]
    vid_dict["ego_viewer_id"] = vid_label[2]
    vid_dict["partner_id"] = vid_label[3]
    vid_dict["location_id"] = vid_label[4]
    vid_dict["activity_id"] = vid_label[5]
    vid_dict["labeled_frames"] = vid_label[6]

    return vid_dict

def convert_Ego_Hand_to_Yolo(EgoHands_dir, EgoHands_yolov8_dir, mode="test"):
    assert mode in ["test", "train"]
    class_id = 0
    mat_file = "/home/inseer/Downloads/egohands_data/metadata.mat"
    mat_data = sio.loadmat(mat_file)["video"]
    mat_data = np.squeeze(mat_data)
    videos = [i for i in range(len(mat_data)) if i > 3] if mode == "train" else [1, 2, 3]
    image_save_dir = Path(EgoHands_yolov8_dir) / f"{mode}/images"
    image_save_dir.mkdir(exist_ok=True, parents=True)
    label_save_dir = Path(EgoHands_yolov8_dir) / f"{mode}/labels"
    label_save_dir.mkdir(exist_ok=True, parents=True)
    ego_frame_dir = Path(EgoHands_dir) / "_LABELLED_SAMPLES"
    for video_num in videos:
        video_labels = convert_vid_label_2_dict(mat_data[video_num])
        print(video_labels["video_id"][0])
        video_path =  ego_frame_dir / video_labels["video_id"][0]
        labeled_frames = np.squeeze(video_labels["labeled_frames"])
        for f_num in range(len(labeled_frames)):
            # copy image over
            frame_id, *hands = labeled_frames[f_num]
            frame_id = int(np.squeeze(frame_id))
            frame_id = add_zeros(frame_id)
            frame_path = (video_path / f"frame_{frame_id}.jpg")
            if not frame_path.is_file():
                print(f"frame {frame_path} doe not exist!")
                continue
            image = cv2.imread(frame_path.as_posix())
            new_frame_name = f"{video_path.name}_{frame_path.name}"
            new_frame_path = image_save_dir / new_frame_name
            shutil.copy(src=str(frame_path), dst=str(new_frame_path))
            # restructure segmentation and save as txt
            txt_file = (label_save_dir / new_frame_path.with_suffix('.txt').name).as_posix()
            with open(txt_file, 'w') as file:
                for hand in hands:
                    if len(hand) > 0:
                        # normalize hand
                        hand = [[round(coord[0] / image.shape[1], 4), round(coord[1] / image.shape[0], 4)] for coord in
                                  hand]
                        hand = np.asarray(hand).flatten().tolist()
                        # Convert the list elements to strings and join them with spaces
                        line = f'{str(class_id)} ' + ' '.join(map(str, hand))
                        line = line + '\n'
                        file.write(line)



if __name__ == "__main__":
    EgoHand_dir = "/home/inseer/Downloads/egohands_data"
    EgoHands_yolov8_dir = "/home/inseer/Downloads/ego_hands_yolov8_crops"
    convert_Ego_Hand_to_Yolo(EgoHand_dir, EgoHands_yolov8_dir, mode="train")
    images_dir = Path(EgoHands_yolov8_dir) / "test/images"
    labels_dir = Path(EgoHands_yolov8_dir) / "test/labels"
    # test_yolo_annotation(images_dir, labels_dir)


