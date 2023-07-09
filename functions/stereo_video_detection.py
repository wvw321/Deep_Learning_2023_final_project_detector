from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import numpy as np
import functions as fn
from test.stereo import video_capture
import time

model = YOLO('../yolov8n.pt')

def detect(path,model):

    cap = video_capture(path, (2560, 720))


    # out = cv2.VideoWriter('output_stereo_video.avi', fourcc, 30.0, (2560, 720))
    out = cv2.VideoWriter('../output_stereo_video_detected.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30,
                          (1280, 720))
    while cap.isOpened():
        start_time = time.time()
        _, frame = cap.read()
        if frame is None:
             break

        imgLeft, imgRight = np.split(frame, 2, axis=1)
        # imgLeft, imgRight = remap(imgLeft, imgRight, stereoMap)
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
        out.write(frame)
        cv2.imshow('YOLO V8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()