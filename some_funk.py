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

def osteochondrosis(image, osteo_switch, osteo_etalon):
    w, h = model_wh('432x368') 
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    max_dif = 0
    num = 0
    main_human = humans[0]
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        shoulder_dif = abs(pose_2d_mpii[2][0] - pose_2d_mpii[5][0])
        if (shoulder_dif > max_dif):
            max_dif = shoulder_dif
            main_human = human
    pose_2d_mpii, visibility = common.MPIIPart.from_coco(main_human)
    if osteo_switch and pose_2d_mpii[2] and pose_2d_mpii[5]:
        osteo_etalon.append((pose_2d_mpii[2][1], pose_2d_mpii[5][1], (pose_2d_mpii[2][0] - pose_2d_mpii[5][0])))
    elif pose_2d_mpii[2] and pose_2d_mpii[5]:
        print(np.divide((pose_2d_mpii[2][1], pose_2d_mpii[5][1], (pose_2d_mpii[2][0] - pose_2d_mpii[5][0])), osteo_etalon))
        num += 1
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
def calibration(image):
    osteo_etalon_array = []
    osteochondrosis(frame, True, osteo_etalon_array)
    return osteo_etalon_array
    
def checker(image, osteo_etalon_array):
    osteo_etalon = np.mean(osteo_etalon_array, axis=0)
    osteochondrosis(frame, False, osteo_etalon)