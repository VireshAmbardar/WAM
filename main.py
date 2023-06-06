## imports
import cv2
import mediapipe as mp

import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)



def main():
    # main code

    # taking input from video
    cap = cv2.VideoCapture("HowSquatsTrim.mp4")

    # running loop on it
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            suc, frame = cap.read()

            
            # // if no frame or user press q 
            if not suc or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

            # // else run below code
            cv2.waitKey(5)
            cv2.imshow("Squats", frame)


        
    return

if __name__ == "__main__":
    main()