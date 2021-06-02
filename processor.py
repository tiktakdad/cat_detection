import numpy as np
import cv2

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
            new_box = [track[2]/10, track[3]/10+track[5]/10/2, track[4]/10, track[5]/10/2]
            new_box = list(map(int, new_box))
            add_map = np.zeros((heat_map.shape[0], heat_map.shape[1]), dtype=np.uint8)
            cv2.rectangle(add_map, new_box, 255, cv2.FILLED)
            filterSize = (int)(new_box[2])
            filterSize = filterSize + 1 if filterSize % 2 == 0 else filterSize
            add_map = cv2.GaussianBlur(add_map, (filterSize, filterSize), 0)
            add_map = (add_map/255)/10
            #cv2.imshow("test", add_map)
            heat_map += add_map
            #print('heat_map_size:' + str(len(head_id)))
            #weighted_img = draw_heatmap(heat_map, stack_frame)

def draw_heatmap(heat_map, stack_frame):
    #upsample_heat_map = cv2.resize(heat_map, (stack_frame.shape[1], stack_frame.shape[0]),interpolation=cv2.INTER_CUBIC)
    heatmapshow = None
    heatmapshow = cv2.normalize(heat_map, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    upsample_heatmapshow = cv2.resize(heatmapshow, (stack_frame.shape[1], stack_frame.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
    mix = 70
    weighted_img = cv2.addWeighted(stack_frame, float(mix) / 100, upsample_heatmapshow, float(100 - mix) / 100, 0)
    return weighted_img