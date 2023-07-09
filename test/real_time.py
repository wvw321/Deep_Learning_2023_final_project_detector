from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np
import functions as fn
from test.stereo import load_stereomap , remap, video_capture
import time

model = YOLO('../yolov8n.pt')
cap = video_capture(numcamera=0, frame_capture_size=(1280, 720), fps=60)
# cap = video_capture("output_stereo.avi",(2560, 720))
stereoMap = load_stereomap("test/stereoMap1280x720.xml")



while True:
    start_time = time.time()
    _, frame = cap.read()
    imgLeft, imgRight = np.split(frame, 2, axis=1)
    imgLeft, imgRight = remap(imgLeft, imgRight, stereoMap)
    imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2RGB)
    imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2RGB)


    resultsL = model.predict(source=imgLeft, show=False, save=False,classes=0,conf=0.7 )
    resultsR = model.predict(source=imgRight, show=False, save=False,classes=0,conf=0.7)

    fn.xywh_to_xcyc(resultsL)
    fn.xywh_to_xcyc(resultsR)
    resultsL = fn.disp(resultsL, resultsR)
    resultsL = fn.depth(resultsL, pixelSize=0.0039, focal=2.8, baseline=120)
    imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2RGB)
    for r in resultsL:

        annotator = Annotator(imgLeft)


        for i,box in enumerate(r):

            b = r.boxes[i].xyxy[0]
            c = r.boxes[i].cls
            dist =r.boxes.dist[i].item()
            annotator.box_label(b, (model.names[int(c)] + str(round(dist,2))))



    frame = annotator.result()
    fps = 1 // (time.time() - start_time)
    fn.draw_fps(img=frame, fps=fps, avg=True)
    cv2.imshow('YOLO V8 Detection', frame)
    cv2.imshow('YOLO V8 Detection1', imgRight)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()