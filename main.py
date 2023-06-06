## imports
import cv2
import mediapipe as mp
import numpy as np

import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

def calculateAngle(point1, point2, point3):
    # Convert points to numpy arrays
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    # Calculate the vectors
    ba = a - b
    bc = c - b

    # Calculate the angles between the vectors
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle)

    return angle_deg



def extractPoints(posepointsDict):
    # // as exercise in in sideways direction
    # guide link https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    # 12 - Right shoulder
    # 11 - left shoulder

    # 24 - right hip
    # 23 - left hip

    # 26 - right knee
    # 25 - left knee

    # 28 - right foot
    # 27 - left foot

    # print(posepointsDict.landmark[12])


    def midPoint(point1, point2):
        x = (point1[0] + point2[0]) / 2
        y = (point1[1] + point2[1]) / 2
        return x,y
                    # //left  - // right
    midShoulder = midPoint((posepointsDict.landmark[12].x,posepointsDict.landmark[12].y),(posepointsDict.landmark[11].x,posepointsDict.landmark[11].y))
    midHip = midPoint((posepointsDict.landmark[24].x,posepointsDict.landmark[24].y),(posepointsDict.landmark[23].x,posepointsDict.landmark[23].y))

    midKnees = midPoint((posepointsDict.landmark[26].x,posepointsDict.landmark[26].y),(posepointsDict.landmark[25].x,posepointsDict.landmark[25].y))

    midFoot = midPoint((posepointsDict.landmark[28].x,posepointsDict.landmark[28].y),(posepointsDict.landmark[27].x,posepointsDict.landmark[27].y))


    # // return only the middle points of shoulder , hips, knees , foot 
    return midShoulder,midHip,midKnees,midFoot

def main():
    # main code

    # taking input from video
    cap = cv2.VideoCapture("HowSquatsTrim.mp4")

    # // function to measure angle between 3 points

    # running loop on it
    counter = 0
    state = "UP"
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            suc, frame = cap.read()

            
            # // if no frame or user press q  = quit
            if not suc or (cv2.waitKey(1) & 0xFF == ord('q')):
                counter = 0 
                break

            # // else run below code

            # // changing it to RGB to increase accuracy
            RGBIMG = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

            # // running media pipe on BGR Image
            RGBIMG.flags.writeable = True

            # // Run pose Estiamtion model on it
            results = pose.process(RGBIMG)

            # // Drawing that images
            mp_drawing.draw_landmarks(RGBIMG, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

           

            # print(results.pose_landmarks)

            # // extract points
            
            midShoulder,midHip,midKnees,midFoot = extractPoints(results.pose_landmarks) 

            ## get angle for the poses
            coreangle = calculateAngle(midShoulder,midHip, midFoot)

            # // if core angle >160 and <200 person is standing constantly
            
            corebendangle = calculateAngle(midShoulder,midHip, midKnees)
            kneeAngle = calculateAngle(midHip,midKnees,midFoot)

            # print(corebendangle , kneeAngle)
            # when both are <90 increase counter by 1

            # when both angles are less than 90 and the state is "UP," increase the counter by 1
            if corebendangle < 90 and kneeAngle < 90 and state == "UP":
                counter += 1
                state = "DOWN"

            # when both angles are greater than 160 and the state is "DOWN," change the state to "UP"
            if corebendangle > 160 and kneeAngle > 160 and state == "DOWN":
                state = "UP"
             
            # // 
            cv2.waitKey(5)
            RGBIMG = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)


            # // text parameters
            orgstate= (50, 200)
            orgcounter = (50,450)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2
            color = (0, 255, 0)  # Green color
            thickness = 2
            lineType = cv2.LINE_AA
            
            # Draw the text on the image
            cv2.putText(RGBIMG, str(counter), orgcounter, fontFace, fontScale, color, thickness, lineType)
            cv2.putText(RGBIMG, str(state), orgstate, fontFace, fontScale, color, thickness, lineType)  
            

            # // converting back to BGR
            
            cv2.imshow("Squats", RGBIMG)
    return

if __name__ == "__main__":
    main()