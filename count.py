import argparse as ap
from collections import defaultdict
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, \
        plot_one_box
from utils.torch_utils import select_device

from sort import Sort


def filter_objects_by_roi(boxes, img, rois):
    for roi in rois:
        slope, intercept, is_upper_plane = roi
        # Draw a line on the resulting image
        xs = np.arange(img.shape[1])
        ys = (slope * xs + intercept).astype(np.int)
        keep = ys < img.shape[0]
        xs = xs[keep]
        ys = ys[keep]
        img[ys, xs, :] = (0, 0, 255)

        # bottom right corners
        xs_br = boxes[:, 0]
        ys_br = boxes[:, 3]
        # filters object lying on the ROI half-plane
        sign = -1 if is_upper_plane else 1
        keep = ys_br < (sign * (slope * xs_br + intercept))
        boxes = boxes[keep]
    return boxes


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--img-size", type=int, default="1280")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--weights", type=str, default="./weights/yolov4-p6.pt")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'Loading weights from {args.weights}')
    model = attempt_load(args.weights, map_location=device)  # load FP32 model

    class_names = model.module.names if hasattr(model, 'module') else model.names
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]

    img_size = 1280
    img_size = check_img_size(img_size, s=model.stride.max())  # check img_size

    vid = cv2.VideoCapture(args.video)
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video size", vw,vh)
        input_fname = os.path.basename(args.video)
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", input_fname.replace(".mp4", "_pred.mp4"))
        out_video = cv2.VideoWriter(output_path, fourcc, 20.0, (vw,vh))

        logs_path = os.path.join("output", input_fname.replace(".mp4", ".csv"))
        logs = []

    # For keeping track of vehicle counts
    class_counts = defaultdict(lambda: 0)
    objects_appeared = set()

    # NOTE: tracker params
    tracker = Sort(max_age=4, min_hits=3, iou_threshold=0.1)

    # NOTE: Change these ROIs for different videos
    # (slope, intercept, is_upper_plane)
    # video: tay_son_input
    # half_planes = [
    #     (0.3, 200, False)
    # ]
    # video: tay_son_output
    half_planes = [
        (0, 200, False),
        (-0.83, 680, False),
    ]

    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=num_frames)
    while True:
        ret, img0 = vid.read()
        if not ret:
            break

        # Preprocess input
        img = letterbox(img0, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        # include only [bicycle, car, motorbike, bus, truck]
        pred = non_max_suppression(pred, 0.5, 0.3, classes=[1, 2, 3, 5, 7], agnostic=False)

        boxes = pred[0]
        boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], img0.shape).round()
        boxes = filter_objects_by_roi(boxes, img0, half_planes)
        boxes = tracker.update(boxes.detach().cpu().numpy())

        for *xyxy, obj_id, cls_id in boxes:
            cls_id = int(cls_id)
            obj_id = int(obj_id)
            # bicycle -> motorbike, truck -> car
            if cls_id == 1:
                cls_id = 3
            elif cls_id == 7:
                cls_id = 2

            c = class_names[cls_id]
            label = f"{c} - {obj_id}"
            plot_one_box(xyxy, img0, label=label, color=colors[cls_id], line_thickness=2)
            if obj_id not in objects_appeared:
                objects_appeared.add(obj_id)
                class_counts[c] += 1
                if args.save_output:
                    logs.append([obj_id, c,
                                 vid.get(cv2.CAP_PROP_POS_MSEC) / 1000])

        count_str = ", ".join(
            [f"{k}: {v}" for k, v in class_counts.items()]
        )
        img0 = cv2.putText(img0, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        img0 = cv2.putText(img0, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if args.save_output:
            out_video.write(img0)
        else:
            cv2.imshow('frame', img0)
            if cv2.waitKey(1) == ord('q'):
                break
        pbar.update(1)

    pbar.close()
    vid.release()
    cv2.destroyAllWindows()
    if args.save_output:
        out_video.release()
        pd.DataFrame(logs).to_csv(logs_path, index=False,
                                  header=["object_id", "class", "timestamp"])
