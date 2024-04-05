import cv2
import os

def extract_frames(video, name):
    """
    extract frames from video source  
    
    Paramaters
    ---------
    video: file (mp4 format)
        a user-recorded, input video
    name: string
        name of the file in string
    """

    #read video with OpenCV VideoCaptrue
    cap = cv2.VideoCapture(video)
    ## video properties
    ret = True 
    while ret:
        ret, frame = cap.read()
        if ret:
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            i = int(msec)
            cv2.imwrite(f"./images/{name}_{i}.jpg", frame)
    cap.release()
