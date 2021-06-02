import argparse

import torch
import numpy as np
import time
import cv2

from tqdm.auto import tqdm
from detector import load_model, get_iou
from processor import process_heatmap, draw_heatmap
from motrackers import IOUTracker
from motrackers.utils import draw_tracks

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

def detection_cat(model, source):
    # capture the webcam
    roi_mul = 1.2
    stacked_box = []
    stacked_id = []
    stacked_time = []
    head_id = {}

    # capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture('D:/Videos/tiki_taka/210601/[mix]TV_CAM_장치_20210601_003220.mp4')
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print('failed capture.isOpened()')
        exit(-1)



    #capture.set(cv2.CAP_PROP_POS_FRAMES, int(19500))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(30))

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(2560))
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(1440))

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heat_map = np.zeros((int(video_height / 10), int(video_width / 10)), dtype=np.float64)

    tracker = IOUTracker(max_lost=15, iou_threshold=0.5, min_detection_confidence=0.4,
                         max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')

    #model, device, stride, imgsz = load_model()

    # Get names and colors
    is_init = True

    #cv2.namedWindow('cam', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('stack_frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    stack_frame = None

    with tqdm(total=frame_count) as pbar:
        while True:  # while true, read the camera
            ret, frame = capture.read()
            if not ret:
                break
            origin = frame.copy()
            if stack_frame is None:
                stack_frame = frame.copy()

            pos_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            pbar.update(pos_frame)

            input = frame[:, :, ::-1]   # OpenCV image (BGR to RGB)
            # torch.stack()

            # Inference
            results = model(input, size=640 , augment=False)
            #results.render()
            #print(results.xyxy[0])
            #print(results.pandas().xyxy[0])

            now = time.localtime()
            n_time = "%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)


            bboxes = []
            confidences = []
            class_ids = []
            for *box, conf, cls in results.pred[0]:  # xyxy, confidence, class
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                # (xmin, ymin, width, height)
                bbox = [c1[0], c1[1], c2[0] - c1[0], c2[1] - c1[1]]
                bbox = expand_box(bbox, roi_mul, frame.shape)

                #label = f'{results.names[int(cls)]} {conf:.2f}'
                need_append_box = True
                for candidate_box in bboxes:
                    if get_iou(candidate_box, bbox) > 0.8:
                        need_append_box = False
                        break

                if need_append_box is True:
                    bboxes.append(bbox)
                    confidences.append(conf.cpu())
                    class_ids.append(cls.cpu())
                    cv2.rectangle(frame, bbox, (0, 0, 255), 3)

            #print(len(bboxes))




            tracks = tracker.update(bboxes, confidences, class_ids)
            process_heatmap(heat_map, tracks, head_id)

            if is_init is True:
                # print(len(stacked_box), len(bboxes))
                inter_idices = []
                for t, track in enumerate(tracks):
                    # cv2.rectangle(frame, (track[2], track[3]), (track[2] + track[4], track[3] + track[5]), (0, 0, 255))
                    # stacked_id.append(track[1])
                    # stacked_box.append([track[2], track[3], track[4], track[5]])

                    dbox = [track[2], track[3], track[4], track[5]]
                    is_intersection = False
                    for sbox in stacked_box:
                        if get_iou(sbox, dbox) > 0.01:
                            is_intersection = True
                            break

                    if is_intersection is False:
                        for sid in stacked_id:
                            if track[1] == sid:
                                is_intersection = True
                                break

                    inter_idices.append(is_intersection)

                for idx, inter in enumerate(inter_idices):
                    if inter is False:
                        new_box = [tracks[idx][2], tracks[idx][3], tracks[idx][4], tracks[idx][5]]

                        # new_size *= roi_mul
                        stacked_box.append(new_box)
                        stacked_id.append(tracks[idx][1])
                        stacked_time.append(n_time)
                        #print(len(stacked_box))


                        stack_frame[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]] \
                            = origin[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]]

                        cv2.imwrite('stack/' + str(len(stacked_box)) + '.png', stack_frame)
                        stack_frame_draw = stack_frame.copy()
                        for s_idx in range(len(stacked_box)):
                            sbox = list(map(int, stacked_box[s_idx]))
                            cv2.rectangle(stack_frame_draw, sbox, (0, 255, 0), 5)
                            cx = (int)(sbox[0] + sbox[2] / 2)
                            cy = (int)(sbox[1] + sbox[3] / 2)
                            cv2.putText(stack_frame_draw, 'ID:' + str(stacked_id[s_idx]), (cx, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(stack_frame_draw, stacked_time[s_idx], (cx - 40, cy + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)

                        cv2.imwrite('stack/' + str(len(stacked_box)) + '_draw' + '.png', stack_frame_draw)
                        heat_map_save = draw_heatmap(heat_map, stack_frame)
                        cv2.imwrite('stack/' + str(len(stacked_box)) + '_heat' + '.png', heat_map_save)
                        #cv2.imshow("heatmap", heat_map_save)
                        # stack_frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


            draw_tracks(frame, tracks)
            if is_init is True:
                for sbox in stacked_box:
                    sbox = list(map(int, sbox))
                    cv2.rectangle(frame, sbox, (0, 255, 0), 5)

                #cv2.imshow('stack_frame', stack_frame)
            #cv2.imshow('cam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # to break the
                break

    # final output
    heat_map_save = draw_heatmap(heat_map, stack_frame)
    cv2.imwrite('stack/' + str(len(stacked_box)) + '_heat' + '.png', heat_map_save)
    #cv2.imshow("heatmap", heat_map_save)
    capture.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/content/drive/MyDrive/mycat/video.mp4', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    detection_cat(model, opt.source)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
