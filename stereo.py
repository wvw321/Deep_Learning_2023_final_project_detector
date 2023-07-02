import cv2

def load_stereomap(filename):
    # Загрузка параметров
    cv_file = cv2.FileStorage()
    cv_file.open(filename, cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y


def remap(imgLeft, imgRight, stereoMap):
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = stereoMap

    imgRight = cv2.remap(imgRight, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgLeft = cv2.remap(imgLeft, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return imgLeft, imgRight

def video_capture(numcamera, frame_capture_size, fps: int = 30):
    cap = cv2.VideoCapture(numcamera)
    while cap.isOpened() == 0:
        print("cap-fail ")
        cap = cv2.VideoCapture(numcamera)

    cap.set(cv2.CAP_PROP_FPS, fps)
    if cap.isOpened() == 0:
        exit(-1)
    W, H = frame_capture_size

    print("FRAME_CAPTURE_SIZE", frame_capture_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    return cap