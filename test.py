from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import functions as fn
from stereo import load_stereomap
import numpy as np




# stereoMap = load_stereomap('stereoMap.xml')




model = YOLO("yolov8n.pt")

frame=cv2.imread("1.png")
imgLeft, imgRight = np.split(frame, 2, axis=1)


resultsL = model.predict(source=imgLeft, show=True, save=True)
resultsR = model.predict(source=imgRight, show=True, save=True)

fn.xywh_to_xcyc(resultsL)
fn.xywh_to_xcyc(resultsR)

resultsL=fn.disp(resultsL, resultsR)

resultsL=fn.depth(resultsL, pixelSize=0.0039, focal=2.8, baseline=120)



print(resultsL)
# for result in results:
#     # Detection
#     # print(result.boxes.xyxy)# box with xyxy format, (N, 4)
#
#
#     result.boxes.xywh # box with xywh format, (N, 4)
#     result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
#     result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
#     result.boxes.conf   # confidence score, (N, 1)
#     result.boxes.cls    # cls, (N, 1)
#
#     print(result.boxes.xy)
#
#     # Classification
#     result.probs     # cls prob, (num_class, )

# cv2.imshow("img", result[0].orig_img)
# cv2.waitKey(0)
