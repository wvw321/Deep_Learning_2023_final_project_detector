
import cv2
import time

import functions as fn
from stereo import load_stereomap , remap, video_capture




cap = video_capture(numcamera=0, frame_capture_size=(1280, 720), fps=30)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (2560, 720))

out = cv2.VideoWriter('output_stereo_video_detected.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30,
                      (2560, 720))

while True:
    _, frame = cap.read()

    # imgLeft, imgRight = np.split(frame, 2, axis=1)
    cv2.imshow('YOLO V8 Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()