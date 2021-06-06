import argparse

import torch
import numpy as np
import cv2
try:
    import google.colab
    from google.colab.patches import cv2_imshow
    is_colab = True
except:
    is_colab = False
print('is_colab:', is_colab)

from detector import load_model, detect_cat
from processor import process_heatmap, process_stackmap, draw_heatmap, save_maps
from motrackers import IOUTracker
from motrackers.utils import draw_tracks


def start_catday(model, source, dest, max_min):
    # (fps * sec * min) / process_time_value
    max_video_length = int((25 * 60 * max_min) / 3)
    roi_mul = 1.2
    stacked_box = []
    stacked_id = []
    head_id = {}

    #capture = cv2.VideoCapture(0)
    #capture = cv2.VideoCapture('D:/Program Files/DAUM/PotPlayer/Capture/TV_CAM_장치_20210526_173127.mp4')
    #capture = cv2.VideoCapture('D:/Videos/tiki_taka/210601/[mix]TV_CAM_장치_20210601_003220.mp4')
    if source == '0':
        source = int(0)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print('failed capture.isOpened()')
        exit(-1)

    # capture.set(cv2.CAP_PROP_POS_FRAMES, int(19500))
    # drop the dummy frames
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(30))

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(2560))
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(1440))

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heat_map = np.zeros((int(video_height / 10), int(video_width / 10)), dtype=np.float64)

    if frame_count > max_video_length:
        skip_frame = int(frame_count / max_video_length)
    else:
        skip_frame = 0
    print('skip frame :', skip_frame)

    tracker = IOUTracker(max_lost=15, iou_threshold=0.5, min_detection_confidence=0.4,
                         max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')

    #cv2.namedWindow('cam', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('stack_frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    stack_frame = None

    while True:  # while true, read the camera
        ret, frame = capture.read()
        if not ret:
            break

        origin_frame = frame.copy()
        if stack_frame is None:
            stack_frame = frame.copy()

        # detection cats
        bboxes, confidences, class_ids = detect_cat(model, frame, roi_mul)
        # track cats
        tracks = tracker.update(bboxes, confidences, class_ids)
        # process stack map
        flag_file_save = process_stackmap(tracks, stacked_box, stacked_id, stack_frame, origin_frame)
        # process heat map
        process_heatmap(heat_map, tracks, head_id)
        # save map files
        if flag_file_save:
            save_maps(dest, stacked_box, stacked_id, stack_frame, heat_map)
        # draw tracks
        draw_tracks(frame, tracks)
        # draw stacked box
        for sbox in stacked_box:
            sbox = list(map(int, sbox))
            cv2.rectangle(frame, sbox, (0, 255, 0), 5)

            # cv2.imshow('stack_frame', stack_frame)
        # cv2.imshow('cam', frame)
        #display(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if is_colab is True:
            cv2_imshow(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # to break the
            break
        if skip_frame > 0:
            now_frame_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(now_frame_pos + skip_frame))


    # final output
    heat_map_save = draw_heatmap(heat_map, stack_frame)
    cv2.imwrite(dest + '/' + str(len(stacked_box)) + '_heat' + '.png', heat_map_save)
    #cv2.imshow("heatmap", heat_map_save)
    capture.release()
    print('Finish Cat Day!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/content/drive/MyDrive/mycat/video.mp4', help='video source')
    parser.add_argument('--dest', type=str, default='/content/drive/MyDrive/mycat/', help='save folder')
    parser.add_argument('--max-minute', type=int, default=10, help='skip frame for max-minute')
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    start_catday(model, opt.source, opt.dest, opt.max_minute)
