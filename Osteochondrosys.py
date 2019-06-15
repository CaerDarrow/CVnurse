'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


#"ON / OFF" variables
ON_OFF_VIDEO = 1
ON_OFF_MUSIK = 1

#OSTEO_SWITCH
SET_OSTEO = True
GET_OSTEO = False 

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.2

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 10

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

def osteochondrosis(image, osteo_switch):
    w, h = model_wh('432x368') 
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
    if osteo_switch and pose_2d_mpii[2] and pose_2d_mpii[5]:
        osteo_etalon.append((pose_2d_mpii[2][1], pose_2d_mpii[5][1], (pose_2d_mpii[2][0] - pose_2d_mpii[5][0])))
    elif pose_2d_mpii[2] and pose_2d_mpii[5]:
        print(np.divide((pose_2d_mpii[2][1], pose_2d_mpii[5][1], (pose_2d_mpii[2][0] - pose_2d_mpii[5][0])), osteo_etalon))
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
        

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368)) 
#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start webcam video capture
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
#time.sleep(2)

osteo_etalon = []
start_time = time.time()
frame_time = 0
while(True):
    #Read each frame and flip it, and convert to grayscale
    seconds = int(frame_time - start_time)
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1) 
    if seconds < 20.0:
        osteochondrosis(frame, SET_OSTEO)
    elif seconds == 20 and len(osteo_etalon) != 3:
        osteo_etalon = np.mean(osteo_etalon, axis=0) 
        print(osteo_etalon)
    elif seconds % 20 == 0:
        osteochondrosis(frame, GET_OSTEO)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Detect facial points through detector function
        faces = detector(gray, 0)
        #Detect faces through haarcascade_frontalface_default.xml
        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

        #Draw rectangle around each face detected
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Detect facial points
        for face in faces:

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            #Get array of coordinates of leftEye and rightEye
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            #Calculate aspect ratio of both eyes
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            #Use hull to remove convex contour discrepencies and draw eye shape around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            #Detect if eye aspect ratio is less than threshold
            if(eyeAspectRatio > EYE_ASPECT_RATIO_THRESHOLD):
                COUNTER += 1
                #If no. of frames is greater than threshold frames,
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    if (ON_OFF_MUSIK != 0):
                        pygame.mixer.music.play(-1)
                    cv2.putText(frame, "Blink please", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            else:
                pygame.mixer.music.stop()
                COUNTER = 0
        
    #Show video feed
    if (ON_OFF_VIDEO != 0):
        cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    frame_time = time.time()

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
