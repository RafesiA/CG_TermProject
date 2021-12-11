import cv2
import cvlib as cv
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

class ShowVideo(QtCore.QObject):
    flag = 0
    grayFlag = 0
    mosaicFlag = 0

    camera = cv2.VideoCapture('Resources/minsu.mp4')

    ret, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal3 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal4 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = 'haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(cascadePath)
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        id = 0

        names = ['None','minsu','junsun', 'kyeong_jin']
        
        minW = 0.1 * self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        minH = 0.1 * self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        global image

        run_video = True
        
        while run_video:
            ret, image = self.camera.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(int(minW), int(minH))
            )
            
            for(x,y,w,h) in faces:
                cv2.rectangle(color_swapped_image, (x,y), (x+w,y+h), (0,255,0),2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 55 :
                    id = names[id]
                else:
                    id = "unknown"
        
                confidence = "  {0}%".format(round(100-confidence))

                cv2.putText(color_swapped_image,str(id), (x+5,y-5),font,1,(255,255,255),2)
                cv2.putText(color_swapped_image,str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)

            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                    self.width,
                                    self.height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)


            if self.flag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_canny = cv2.Canny(img_gray, 50, 100)

                qt_image2 = QtGui.QImage(img_canny.data,
                                         self.width,
                                         self.height,
                                         img_canny.strides[0],
                                         QtGui.QImage.Format_Grayscale8)
                
                self.VideoSignal2.emit(qt_image2)
                
            if self.grayFlag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                qt_image3 = QtGui.QImage(img_gray.data,
                                         self.width,
                                         self.height,
                                         img_gray.strides[0],
                                         QtGui.QImage.Format_Grayscale8)
                
                self.VideoSignal3.emit(qt_image3)
            if self.mosaicFlag:
                img_mosaic = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for(x,y,w,h) in faces:
                    face_img = img_mosaic[y:y+h, x:x+w] # 탐지된 얼굴 이미지 crop
                    face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # 축소
                    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) # 확대
                    img_mosaic[y:y+h, x:x+w] = face_img # 탐지된 얼굴 영역 모자이크 처리
                    
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    
                        

                qt_image4 = QtGui.QImage(img_mosaic.data,
                                             self.width,
                                             self.height,
                                             img_mosaic.strides[0],
                                             QtGui.QImage.Format_RGB888)
                self.VideoSignal4.emit(qt_image4)
                
                

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit) #25 ms
            loop.exec_()

    @QtCore.pyqtSlot()
    def canny(self):
        self.flag = 1 - self.flag
    
    @QtCore.pyqtSlot()
    def gray(self):
        self.grayFlag = 1 - self.grayFlag
        
    
    @QtCore.pyqtSlot()
    def blur(self):
        self.blurFlag = 1 - self.blurFlag
        
    @QtCore.pyqtSlot()
    def mosaic(self):
        self.mosaicFlag = 1 - self.mosaicFlag


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)


    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    image_viewer2 = ImageViewer()
    image_viewer3 = ImageViewer()
    image_viewer4 = ImageViewer()

    vid.VideoSignal1.connect(image_viewer1.setImage)
    vid.VideoSignal2.connect(image_viewer2.setImage)
    vid.VideoSignal3.connect(image_viewer3.setImage)
    vid.VideoSignal4.connect(image_viewer4.setImage)

    push_button1 = QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Canny')
    push_button3 = QtWidgets.QPushButton('Gray')
    push_button4 = QtWidgets.QPushButton('Mosaic')
    
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.canny)
    push_button3.clicked.connect(vid.gray)
    push_button4.clicked.connect(vid.mosaic)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    
    horizontal_layout.addWidget(image_viewer1)
    horizontal_layout.addWidget(image_viewer2)
    horizontal_layout.addWidget(image_viewer3)
    horizontal_layout.addWidget(image_viewer4)
    
    vertical_layout.addLayout(horizontal_layout)
    
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)
    vertical_layout.addWidget(push_button3)
    vertical_layout.addWidget(push_button4)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    main_window.setWindowTitle('얼굴 인식 및 변조 프로그램')
    sys.exit(app.exec_())