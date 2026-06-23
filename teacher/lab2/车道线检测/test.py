import cv2
path = r"C:\ZC\HIT\Compulsory_Courses\CV\CV_lab\teacher\lab2\outputs\output_compare.mp4"
cap = cv2.VideoCapture(path)
print("opened:", cap.isOpened())
print("frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("width,height:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps:", cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
print("read first frame:", ret)
if ret:
    print("frame dtype, shape:", frame.dtype, frame.shape)
cap.release()