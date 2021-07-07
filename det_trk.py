import torch
import cv2
from detector import load_model, detect_cat
from motrackers import IOUTracker
from motrackers.utils import draw_tracks


def det(model):
    roi_mul = 1.2
    color = [(186, 72, 0), (0, 0, 186)]
    sub_color = [(232, 185, 124), (111, 111, 232)]
    # capture = cv2.VideoCapture(0)
    # capture = cv2.VideoCapture('D:/Program Files/DAUM/PotPlayer/Capture/TV_CAM_장치_20210526_173127.mp4')
    # capture = cv2.VideoCapture('D:/Videos/tiki_taka/210601/[mix]TV_CAM_장치_20210601_003220.mp4')
    lt_rb = [1135, 65, 1190, 915]
    frames = []
    b_bboxes = []
    b_tracks = [[], []]
    frame = cv2.imread('D:/Pictures/tikitaka/DSC04099.jpg', cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))

    tracker = IOUTracker(max_lost=15, iou_threshold=0.5, min_detection_confidence=0.4,
                         max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')
    # cv2.namedWindow('cam', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    line_cnt = 7

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

    # draw_tracks(frame, tracks)

    # draw stacked box
    text_w = 160
    text_h = 40
    gt = [275, 620, 640-275, 1300-620]
    steps = [[+80, 80, 40, -40, 48], [-20, -20, 100, -50, 65], [-40, 60, -30, -120, 77], [0, 0, 0, 0, 92], [0, 0, 0, 0, 99]]
    for i, step in enumerate(steps):
        for bbox in bboxes:
            cx = int(bbox[0] + (bbox[2]) / 2)
            cy = int(bbox[1] + (bbox[3]) / 2)
            c_bbox = bbox.copy()
            c_bbox[0] += step[0] * 2
            c_bbox[1] += step[1] * 2
            c_bbox[2] += step[2] * 2
            c_bbox[3] += step[3] * 2
            if i == len(steps)-1:
                c_bbox = gt[0:4]

            new_box_frame = box_frame.copy()
            '''
            cv2.rectangle(new_box_frame, gt, (32, 23, 245), thickness=8, lineType=cv2.LINE_AA)
            cv2.rectangle(new_box_frame, c_bbox, (43, 186, 0), thickness=8, lineType=cv2.LINE_AA)
            cv2.circle(new_box_frame, (cx, cy), 4, (43, 186, 0), thickness=8,
                       lineType=cv2.LINE_AA)


            cv2.arrowedLine(new_box_frame, (c_bbox[0], c_bbox[1]), (gt[0], gt[1]), (43, 186, 0), thickness=6,
                            line_type=cv2.LINE_AA)
            cv2.arrowedLine(new_box_frame, (c_bbox[0]+c_bbox[2], c_bbox[1]), (gt[0]+gt[2], gt[1]), (43, 186, 0), thickness=6,
                            line_type=cv2.LINE_AA)
            cv2.arrowedLine(new_box_frame, (c_bbox[0], c_bbox[1]+c_bbox[3]), (gt[0], gt[1]+gt[3]), (43, 186, 0), thickness=6,
                            line_type=cv2.LINE_AA)
            cv2.arrowedLine(new_box_frame, (c_bbox[0]+c_bbox[2], c_bbox[1]+c_bbox[3]), (gt[0]+gt[2], gt[1]+gt[3]), (43, 186, 0), thickness=6,
                            line_type=cv2.LINE_AA)
            cv2.rectangle(new_box_frame, (int(c_bbox[0] - 3), int(c_bbox[1] - text_h), text_w, text_h), (43, 186, 0),
                          thickness=cv2.FILLED)

            cv2.putText(new_box_frame, 'CAT: ' + str(step[4]) + '%', (int(c_bbox[0]), int(c_bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (215, 215, 215), 3, lineType=cv2.LINE_AA)
            '''
            cv2.imshow('cam', new_box_frame)
            cv2.imwrite('steps_' + str(i) + '.png', new_box_frame)
            cv2.waitKey(1)

    # cv2.imwrite('box_' + str(f) + '.jpg', box_frame)





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

        for track in tracks:
            cv2.rectangle(frame, (track[2], track[3]), (track[2] + track[4], track[3] + track[5]), color[int(track[1])], thickness=5, lineType=cv2.LINE_AA)
        cv2.imwrite('track_' + str(f) + '.jpg', frame)


        #draw_tracks(frame, tracks)

        # draw stacked box

        if cv2.waitKey(0) & 0xFF == ord('q'):  # to break the
            break


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device, 0.10, 0.35, [15, 16])
    det(model)
