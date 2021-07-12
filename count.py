import os
import argparse as ap
import cv2
import numpy as np
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size
import torch

from sort import Sort


def draw_line(img, slope=1, intercept=0):
    xs = np.arange(img.shape[1])
    ys = (slope * xs + intercept).astype(np.int)
    keep = ys < img.shape[0]
    xs = xs[keep]
    ys = ys[keep]
    img[ys, xs, :] = (0, 0, 255)


def filter_objects_by_line(boxes, img, slope=1, intercept=0):
    width = img.shape[1]
    height = img.shape[0]
    # bottom right corners
    xs_br = boxes[:, 0] * width
    ys_br = boxes[:, 3] * height
    keep = ys_br > (slope * xs_br + intercept).astype(np.int)
    return boxes[keep]


def filter_objects_by_class(boxes):
    # [bicycle, car, motorbike, bus, truck]
    included_classes = [1, 2, 3, 5, 7]
    keep = np.isin(boxes[:, 5].astype(np.int), included_classes)
    return boxes[keep]


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--cfg", type=str, default="./cfg/yolov4-tiny.cfg")
    parser.add_argument("--weights", type=str, default="./weights/yolov4-tiny.weights")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    # print(f'Loading model config from {args.cfg}')
    print(f'Loading weights from {args.weights}')
    model = attempt_load(args.weights, map_location=device)  # load FP32 model

    img_size = 640
    img_size = check_img_size(img_size, s=model.stride.max())  # check img_size

    class_names = model.module.names if hasattr(model, 'module') else model.names

    tracker = Sort(max_age=1000, min_hits=3)
    vid = cv2.VideoCapture(args.video)
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video size", vw,vh)
        input_fname = os.path.basename(args.video)
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", input_fname.replace(".mp4", "_pred.mp4"))
        out_video = cv2.VideoWriter(output_path, fourcc, 20.0, (vw,vh))

    # NOTE: Change these for different videos
    slope = 0.3
    intercept = 200

    pbar = tqdm(total=num_frames)
    while True:
        ret, img0 = vid.read()
        if not ret:
            break

        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        print(pred)
        break

        # boxes = filter_objects_by_class(np.array(boxes))
        # boxes = filter_objects_by_line(boxes, frame, slope, intercept)
        # tracked_boxes = tracker.update(boxes)

        # res_img = plot_boxes_cv2(frame, tracked_boxes, class_names=class_names)
        # draw_line(res_img, slope, intercept)
        # count_str = ", ".join(
        #     [f"{class_names[k]}: {v}" for k, v in tracker.object_counts.items()]
        # )
        # res_img = cv2.putText(res_img, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        # res_img = cv2.putText(res_img, count_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # if args.save_output:
        #     out_video.write(res_img)
        # else:
        #     cv2.imshow('frame', res_img)
        #     if cv2.waitKey(1) == ord('q'):
        #         break
        # pbar.update(1)

    pbar.close()
    vid.release()
    cv2.destroyAllWindows()
    if args.save_output:
        out_video.release()
