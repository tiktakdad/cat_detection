import numpy as np
import cv2
from detector import get_iou


def save_maps(dest, stacked_box, stacked_id, stack_frame, heat_map):
    cv2.imwrite(dest + '/' + str(len(stacked_box)) + '.png', stack_frame)
    stack_frame_draw = stack_frame.copy()
    for s_idx in range(len(stacked_box)):
        sbox = list(map(int, stacked_box[s_idx]))
        cv2.rectangle(stack_frame_draw, sbox, (0, 255, 0), 5)
        cx = (int)(sbox[0] + sbox[2] / 2)
        cy = (int)(sbox[1] + sbox[3] / 2)
        cv2.putText(stack_frame_draw, 'ID:' + str(stacked_id[s_idx]), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(dest + '/' + str(len(stacked_box)) + '_draw' + '.png', stack_frame_draw)
    heat_map_save = draw_heatmap(heat_map, stack_frame)
    cv2.imwrite(dest + '/' + str(len(stacked_box)) + '_heat' + '.png', heat_map_save)


def process_stackmap(tracks, stacked_box, stacked_id, stack_frame, origin_frame):
    # print(len(stacked_box), len(bboxes))
    inter_idices = []
    flag_file_save = False
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
            flag_file_save = True
            new_box = [tracks[idx][2], tracks[idx][3], tracks[idx][4], tracks[idx][5]]

            # new_size *= roi_mul
            stacked_box.append(new_box)
            stacked_id.append(tracks[idx][1])
            # print(len(stacked_box))

            stack_frame[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]] \
                = origin_frame[new_box[1]:new_box[1] + new_box[3], new_box[0]:new_box[0] + new_box[2]]

    return flag_file_save


def process_heatmap(heat_map, tracks, head_id):
    weighted_img = None
    for t, track in enumerate(tracks):
        need_add = False
        if head_id.get(track[1]) is None:
            head_id[track[1]] = 1
            need_add = True
        else:
            if head_id[track[1]] > 120:
                need_add = False
            else:
                head_id[track[1]] += 1
                need_add = True

        if need_add == True:
            new_box = [track[2] / 10, track[3] / 10 + track[5] / 10 / 2, track[4] / 10, track[5] / 10 / 2]
            new_box = list(map(int, new_box))
            add_map = np.zeros((heat_map.shape[0], heat_map.shape[1]), dtype=np.uint8)
            cv2.rectangle(add_map, new_box, 255, cv2.FILLED)
            filterSize = (int)(new_box[2])
            filterSize = filterSize + 1 if filterSize % 2 == 0 else filterSize
            add_map = cv2.GaussianBlur(add_map, (filterSize, filterSize), 0)
            add_map = (add_map / 255) / 10
            # cv2.imshow("test", add_map)
            heat_map += add_map
            # print('heat_map_size:' + str(len(head_id)))
            # weighted_img = draw_heatmap(heat_map, stack_frame)


def draw_heatmap(heat_map, stack_frame):
    # upsample_heat_map = cv2.resize(heat_map, (stack_frame.shape[1], stack_frame.shape[0]),interpolation=cv2.INTER_CUBIC)
    heatmapshow = None
    heatmapshow = cv2.normalize(heat_map, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    upsample_heatmapshow = cv2.resize(heatmapshow, (stack_frame.shape[1], stack_frame.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)
    mix = 70
    weighted_img = cv2.addWeighted(stack_frame, float(mix) / 100, upsample_heatmapshow, float(100 - mix) / 100, 0)
    return weighted_img
