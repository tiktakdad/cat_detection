import torch
import cv2
import numpy as np


def load_model(device, conf, iou, classes):
    # device = select_device('0')

    # model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = conf  # confidence threshold (0-1)
    model.iou = iou  # NMS IoU threshold (0-1)
    model.classes = classes
    # model.classes = [0, 15, 16]   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.to(device)

    return model

def detect_cat(model, frame, roi_mul):

    input = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
    results = model(input, size=640, augment=False)
    bboxes = []
    confidences = []
    class_ids = []
    for *box, conf, cls in results.pred[0]:  # xyxy, confidence, class
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # (xmin, ymin, width, height)
        bbox = [c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1]]
        bbox = expand_box(bbox, roi_mul, frame.shape)

        # label = f'{results.names[int(cls)]} {conf:.2f}'
        need_append_box = True
        for candidate_box in bboxes:
            if get_iou(candidate_box, bbox) > 0.8:
                need_append_box = False
                break

        if need_append_box is True:
            bboxes.append(bbox)
            confidences.append(conf.cpu())
            class_ids.append(cls.cpu())
            #cv2.rectangle(frame, bbox, (0, 0, 255), 3)
    return bboxes, confidences, class_ids

def get_iou(bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[0]+bb1[2] - bb1[0]) * (bb1[1]+bb1[3] - bb1[1])
    bb2_area = (bb2[0]+bb2[2] - bb1[0]) * (bb2[1]+bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def transpose_img_to_input(frame, imgsz, stride, device):
    img = letterbox(frame, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def expand_box(bbox, roi_mul, shape):
    cx = int(bbox[0] + bbox[2] / 2)
    cy = int(bbox[1] + bbox[3] / 2)
    width = int(bbox[2] * roi_mul)
    height = int(bbox[3] * roi_mul)
    # expand the box
    cx = cx if cx - int(width / 2) > 0 else cx + abs(cx - int(width / 2))
    cy = cy if cy - int(height / 2) > 0 else cy + abs(cy - int(height / 2))
    cx = cx if cx + int(width / 2) < shape[1] else cx - abs(
        (cx + int(width / 2)) - shape[1])
    cy = cy if cy + int(height / 2) < shape[0] else cy - abs(
        (cy + int(height / 2)) - shape[0])
    new_box = [cx - int(width / 2), cy - int(height / 2), width, height]
    return new_box