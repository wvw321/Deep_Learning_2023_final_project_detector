import cv2
import numpy as np
from test.stereo import load_stereomap , remap, video_capture




cap = video_capture("output.avi",(2560, 720))

# out = cv2.VideoWriter('output_stereo.avi', fourcc, 30.0, (2560, 720))
out = cv2.VideoWriter('../output_stereo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30,
                      (2560, 720))
stereoMap = load_stereomap("stereoMap1280x720.xml")
while True:

    _, frame = cap.read()

    imgLeft, imgRight = np.split(frame, 2, axis=1)
    imgLeft, imgRight = remap(imgLeft, imgRight, stereoMap)
    frame=np.hstack((imgLeft, imgRight))

    cv2.imshow('YOLO V8 Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
# out.release()
cv2.destroyAllWindows()