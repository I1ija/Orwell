import cv2
vidcap = cv2.VideoCapture('ilija_.mp4')
success,image = vidcap.read()
count = 0
while success:
    if count%1==0:
        cv2.imwrite("faces/ilija_%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

    

