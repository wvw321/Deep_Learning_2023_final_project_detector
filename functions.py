import torch
import cv2

def xywh_to_xcyc(results):
    for i, result in enumerate(results):
        xy = []
        boxes = result.boxes.xywh
        for xywh in boxes:
            x = xywh[0]
            y = xywh[1]
            w = xywh[2]
            h = xywh[3]
            xy.append([x + w / 2, y + h / 2])
        results[i].boxes.xy = torch.tensor(xy)


def depth(results, pixelSize, focal, baseline):
    for i, box in enumerate(results):
        depth_list = []
        for j, _ in enumerate(results[i]):
            disparity = box.boxes.disp[j].item()
            if  disparity >0:
                depth = (baseline * focal) / (disparity * pixelSize * 1000)
            else:
                depth=0

            depth_list.append(depth)
        results[i].boxes.dist = torch.tensor(depth_list).unsqueeze(1)
    return results


def disp(resultsL, resultsR):
    for i, _ in enumerate(resultsL):
        dispp = []
        boxesL = resultsL[i].boxes.xy
        boxesR = resultsR[i].boxes.xy
        for j, xyR in enumerate(boxesL):
            xl, yl = boxesL[j][0], boxesL[j][1]
            try:
                xr, yr = boxesR[j][0], boxesR[j][1]
                dispp.append(((xl - xr) ** 2 + (yl - yr) ** 2) ** 0.5)
            except:
                dispp.append(0)

        resultsL[i].boxes.disp = torch.tensor(dispp).unsqueeze(1)

    return resultsL


def draw_fps(img, fps, avg=False, list=[]):
    if avg is False:
        cv2.putText(img, "fps " + str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        return
    if avg is True:
        list.append(fps)
        if len(list) > 30:
            list.pop(0)
        avg_fps = sum(list) // len(list)
        cv2.putText(img, "AVG_Fps" + str(avg_fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)