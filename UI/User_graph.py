import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
from ultralytics import YOLO

# 界面设计
class AnimalDetectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = YOLO(r'D:\Python\graduate_design\ultralytics-8.3.2\yolo11n-seg.pt')

    def initUI(self):
        self.setWindowTitle('Animal Detection System')
        self.layout = QVBoxLayout()
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.button = QPushButton('Open Image or Video', self)
        self.button.clicked.connect(self.open_file)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                   "All Files (*);;MP4 Files (*.mp4);;JPEG Files (*.jpg);;PNG Files (*.png)",
                                                   options=options)
        if file_path:
            if file_path.endswith('.mp4'):
                self.detect_animal_video(file_path)
            else:
                self.detect_animal_image(file_path)

    def detect_animal_image(self, file_path):
        frame = cv2.imread(file_path)
        results = self.model(frame)
        for result in results:
            bbox = result['bbox']
            label = result['label']
            confidence = result['confidence']

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def detect_animal_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

        results = self.model(frame)
        for result in results:
            bbox = result['bbox']
            label = result['label']
            confidence = result['confidence']

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                2)

        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(qImg))
        cv2.waitKey(1)

        cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnimalDetectionUI()
    ex.show()
    sys.exit(app.exec_())
