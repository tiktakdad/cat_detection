import torch
import cv2
from detector import load_model, detect_cat
from motrackers import IOUTracker
from motrackers.utils import draw_tracks


def det_trk(model):
    roi_mul = 1.2
    color = [(186,72,0), (0,0,186)]
    sub_color = [(232,185,124), (111,111,232)]
    # capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture('D:/Program Files/DAUM/PotPlayer/Capture/TV_CAM_장치_20210526_173127.mp4')
    # capture = cv2.VideoCapture('D:/Videos/tiki_taka/210601/[mix]TV_CAM_장치_20210601_003220.mp4')
    lt_rb = [1135, 65, 1190, 915]
    frames = []
    b_bboxes = []
    b_tracks = [[],[]]
    for i in range(0,4):
        fn = 'D:/Pictures/tikitaka/track/'+str(i)+'.jpg'
        frames.append(cv2.imread(fn, cv2.IMREAD_COLOR)[lt_rb[1]:lt_rb[0], lt_rb[1]+lt_rb[3]:lt_rb[0]+lt_rb[2]])

    tracker = IOUTracker(max_lost=15, iou_threshold=0.5, min_detection_confidence=0.4,
                         max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')
    #cv2.namedWindow('cam', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    line_cnt = 18

    for f, frame in enumerate(frames):
        # detection cats
        draw_frame = frame.copy()
        box_frame = frame.copy()
        bboxes, confidences, class_ids = detect_cat(model, frame, roi_mul)
        # track cats
        tracks = tracker.update(bboxes, confidences, class_ids)

        cell_h = int(frame.shape[0] / line_cnt)
        cell_w = int(frame.shape[1] / line_cnt)
        for l in range(0, line_cnt):
            cv2.line(box_frame, (0, (l + 1) * cell_h), (frame.shape[1], (l + 1) * cell_h), (255, 255, 255), thickness=1,
                     lineType=cv2.LINE_AA)
        for l in range(0, line_cnt):
            cv2.line(box_frame, ((l + 1) * cell_w, 0), ((l + 1) * cell_w, frame.shape[0]), (255, 255, 255), thickness=1,
                     lineType=cv2.LINE_AA)

        for bbox in bboxes:
            cx = int(bbox[0] + (bbox[2]) / 2)
            cy = int(bbox[1] + (bbox[3]) / 2)
            cv2.rectangle(box_frame, bbox, (43, 186, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.circle(box_frame, (cx, cy), 4, (43, 186, 0), thickness=4,
                       lineType=cv2.LINE_AA)
        cv2.imwrite('box_' + str(f) + '.jpg', box_frame)

        for b_boxes in b_bboxes:
            cv2.rectangle(draw_frame, b_boxes, (121,232,150), thickness=4, lineType=cv2.LINE_AA)

        b_bboxes.clear()
        for bbox in bboxes:
            cx = int(bbox[0] + (bbox[2]) / 2)
            cy = int(bbox[1] + (bbox[3]) / 2)
            cv2.rectangle(draw_frame, bbox, (43, 186, 0), thickness=4, lineType=cv2.LINE_AA)
            cv2.circle(draw_frame, (cx, cy), 4, (43, 186, 0), thickness=4,
                     lineType=cv2.LINE_AA)
            b_bboxes.append(bbox)
        cv2.imwrite('detect_'+str(f)+'.jpg', draw_frame)


        for track in tracks:
            b_tracks[int(track[1])].append(track)
        # draw tracks
        for j in range(0,2):
            for k in range(0, len(b_tracks[j])):
                b_cx = int(b_tracks[j][k][2] + (b_tracks[j][k][4]) / 2)
                b_cy = int(b_tracks[j][k][3] + (b_tracks[j][k][5]) / 2)
                if k > 0:
                    cx = int(b_tracks[j][k - 1][2] + (b_tracks[j][k - 1][4]) / 2)
                    cy = int(b_tracks[j][k - 1][3] + (b_tracks[j][k - 1][5]) / 2)
                else:
                    cx = b_cx
                    cy = b_cy

                cv2.line(frame, (b_cx, b_cy), (cx, cy), color[int(b_tracks[j][k][1])], thickness=5,
                         lineType=cv2.LINE_AA)

                if k == len(b_tracks[j]) - 1:
                    cv2.rectangle(frame, (b_tracks[j][k][2], b_tracks[j][k][3]),
                                  (b_tracks[j][k][2] + b_tracks[j][k][4], b_tracks[j][k][3] + b_tracks[j][k][5]),
                                  color[int(b_tracks[j][k][1])],
                                  thickness=5, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (b_cx, b_cy), 4,  color[int(b_tracks[j][k][1])], thickness=4,
                               lineType=cv2.LINE_AA)
                elif k == len(b_tracks[j]) - 2:
                    cv2.rectangle(frame, (b_tracks[j][k][2], b_tracks[j][k][3]),
                                  (b_tracks[j][k][2] + b_tracks[j][k][4], b_tracks[j][k][3] + b_tracks[j][k][5]),
                                  sub_color[int(b_tracks[j][k][1])],
                                  thickness=5, lineType=cv2.LINE_AA)

        '''
        for track in tracks:
            cv2.rectangle(frame, (track[2], track[3]), (track[2] + track[4], track[3] + track[5]), color[int(track[1])], thickness=5, lineType=cv2.LINE_AA)
            '''
        cv2.imwrite('track_' + str(f) + '.jpg', frame)


        #draw_tracks(frame, tracks)

        # draw stacked box

        #cv2.imshow('cam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # to break the
            break


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device, 0.10, 0.35, [15, 16])
    det_trk(model)
