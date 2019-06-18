'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
import time
import dlib
import cv2
import sys
import pync
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QCheckBox, QMessageBox
from PyQt5.QtMultimedia import QSound


class Spine_class():
    def __init__(self):
        self.humans = 0
        self.osteo_etalon = []
        self.calibrate = False
        self.calibrated = False
        self.call = False
        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))   
        
    def add(self, pose_2d_mpii):
        self.osteo_etalon.append((pose_2d_mpii[2][1], pose_2d_mpii[5][1], abs(pose_2d_mpii[5][0] - pose_2d_mpii[2][0]), abs(pose_2d_mpii[0][1] - pose_2d_mpii[1][1])))
        
    def get(self, pose_2d_mpii):
        estim = np.divide((pose_2d_mpii[2][1], pose_2d_mpii[5][1], abs(pose_2d_mpii[5][0] - pose_2d_mpii[2][0]), abs(pose_2d_mpii[0][1] - pose_2d_mpii[1][1])), self.osteo_etalon)
        print(estim)
        if (abs(pose_2d_mpii[5][1] - pose_2d_mpii[2][1]) > 0.05):
            pync.notify("Correct the back!!!")
        elif ((estim[2]/estim[3]) > 1.1):
            pync.notify("Correct the back!!!")
        elif (estim[2] > 1.1):
            pync.notify("Go far from the screen!!!")
        
    def __call__(self, image):
        if self.calibrate:
            self.osteochondrosis(image, self.add)
        elif self.calibrated and not self.calibrate and len(self.osteo_etalon) != 4:
            self.osteo_etalon = np.mean(self.osteo_etalon, axis=0)
            print(self.osteo_etalon)
        elif self.calibrated and self.call:
            self.osteochondrosis(image, self.get)
            
    def osteochondrosis(self, image, func):
        w, h = model_wh('432x368')
        try:
            humans = self.e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            max_dif = 0
            main_human = humans[0]
            for human in humans:
                pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                shoulder_dif = abs(pose_2d_mpii[2][0] - pose_2d_mpii[5][0])
                if (shoulder_dif > max_dif):
                    max_dif = shoulder_dif
                    main_human = human
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(main_human)
            func(pose_2d_mpii)
            image = TfPoseEstimator.draw_humans(image, [main_human], imgcopy=False)
        except:
            pass
    
class Eye_class(): 
    #Minimum threshold of eye aspect ratio below which alarm is triggerd
    EYE_ASPECT_RATIO_THRESHOLD = 0.25
    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 6
    #COunts no. of consecutuve frames below threshold value
    COUNTER = time.time()
    def __init__(self):
        #Load face cascade which will be used to draw a rectangle around detected faces.
        self.face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
         #Load face detector and predictor, uses dlib shape predictor file
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        #Extract indexes of facial landmarks for the left and right eye
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        self.ON_OFF_VIDEO = 1
        self.ON_OFF_MUSIK = 1
        self.NOTIFICATION = 1   
    
    #This function calculates and return eye aspect ratio
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A+B) / (2*C)
        return ear
    
    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Detect facial points through detector function
        faces = self.detector(gray, 0)
        #Detect faces through haarcascade_frontalface_default.xml
        face_rectangle = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        #Draw rectangle around each face detected
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #Detect facial points
        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            #Get array of coordinates of leftEye and rightEye
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            #Calculate aspect ratio of both eyes
            leftEyeAspectRatio = self.eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = self.eye_aspect_ratio(rightEye)
            self.eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
            #Use hull to remove convex contour discrepencies and draw eye shape around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            #Detect if eye aspect ratio is less than threshold
            if(self.eyeAspectRatio > self.EYE_ASPECT_RATIO_THRESHOLD):
                #If no. of frames is greater than threshold frames,
                if time.time() - self.COUNTER >= self.EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    
                    if (self.ON_OFF_MUSIK != 0):
                        QSound.play('audio/alert.wav')
                    start_alarm = time.time()
                    cv2.putText(frame, "Blink please", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    while (time.time() - start_alarm < 0.4):
                        pass
                    self.COUNTER = time.time()
                    if self.NOTIFICATION != 0:
                        pync.notify('Blink please')

            else:
                self.COUNTER = time.time()
        return face_rectangle

class FaceDetectionWidget(QtWidgets.QWidget):
    image_data = QtCore.pyqtSignal(np.ndarray)  
    def __init__(self, parent, camera_port=0):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier()
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.eye = Eye_class()
        self.spine = Spine_class()
        self.camera = cv2.VideoCapture(camera_port)
        self.timer_recording = QtCore.QTimer()
        self.timer_spine = QtCore.QTimer()
        self_timer_eye = QtCore.QTimer()
        
        
    def detect_faces(self, image: np.ndarray): 
        self.check_spine()    
        self.spine(image)
        self.spine.call = False
        if(not self.spine.call):
            faces_rec = self.eye(image)
        return faces_rec
    
    def start_calibrate(self):
        msg = QMessageBox()
        msg.setText("Please sit right")
        msg.setWindowTitle("Information Window")
        msg.exec_()
        self.spine.calibrate = True
        QtCore.QTimer.singleShot(8000, self.stop_calibrate)
    
    def check_spine(self):
        if self.spine.calibrated:
            if not self.timer_spine.isActive():
                self.timer_spine.start(10000)
                self.timer_spine.timeout.connect(self.call_switch)
        
    def stop_calibrate(self):
        self.spine.calibrate = False
        self.spine.calibrated = True
        msg = QMessageBox()
        msg.setText("Calibration Finished")
        msg.setWindowTitle("Information Window")
        msg.exec_()
       
    def call_switch(self):
        self.spine.call = True
        
    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._red,
                          self._width)
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage
        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def start_recording(self):
        self.timer_recording.start(3)
        self.timer_recording.timeout.connect(self.read_frame)
    
    def read_frame(self):
        read, data = self.camera.read()
        cv2.flip(data, 0)
        if read:
            self.image_data.emit(data)

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_detection_widget = FaceDetectionWidget(self)
        image_data_slot = self.face_detection_widget.image_data_slot
        self.face_detection_widget.image_data.connect(image_data_slot)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        self.run_button2 = QtWidgets.QPushButton('Calibrate')
        self.checkbox = QCheckBox("Sound ON", self)
        self.checkbox.setChecked(True)
        self.checkbox2 = QCheckBox("Notify", self)
        self.checkbox2.setChecked(True)
        layout.addWidget(self.run_button)
        layout.addWidget(self.run_button2)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.checkbox2)
        self.run_button2.clicked.connect(self.face_detection_widget.start_calibrate)
        self.run_button.clicked.connect(self.face_detection_widget.start_recording)
        self.checkbox.stateChanged.connect(self.clickSound)
        self.checkbox2.stateChanged.connect(self.clickNotification)
        self.setLayout(layout)

    def clickSound(self, state):
        self.face_detection_widget.eye.ON_OFF_MUSIK = 1 if state == QtCore.Qt.Checked else 0

    def clickNotification(self, state):
        self.face_detection_widget.eye.NOTIFICATION = 1 if state == QtCore.Qt.Checked else 0


def main():    
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
 
