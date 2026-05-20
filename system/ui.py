#Written by Kaiden Hu @2025.05
#This is UI of the mobile screen defect detection system.
import os
import sys
from pathlib import Path

# Ensure Qt plugins are loaded from the current conda environment's PyQt5 installation.
qt_plugins_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "PyQt5" / "Qt5" / "plugins"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_plugins_path)
os.environ["QT_PLUGIN_PATH"] = str(qt_plugins_path)

import cv2

# cv2 may override QT_QPA_PLATFORM_PLUGIN_PATH during import, restore it here.
if qt_plugins_path.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_plugins_path)
    os.environ["QT_PLUGIN_PATH"] = str(qt_plugins_path)
else:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)

from PyQt5.QtCore import QThread, Qt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
import torch
from ultralytics import YOLO

class DefectDetector(QThread):
    detection_done = QtCore.pyqtSignal(object, str)
    
    def __init__(self, image_path, model, names_map):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.names_map = names_map
    
    def run(self):
        try:
            # 加载图像
            img = cv2.imread(self.image_path)
            if img is None:
                self.detection_done.emit(None, "无法读取图像")
                return
            
            # 进行检测
            results = self.model(img)
            
            # 提取检测信息，并过滤低置信度结果
            detections = results[0].boxes
            filtered_boxes = [box for box in detections if box.conf.item() > 0.5]
            defect_info = f"检测到 {len(filtered_boxes)} 个缺陷:\n"
            annotated_img = img.copy()

            for i, box in enumerate(filtered_boxes):
                cls = int(box.cls.item())
                conf = box.conf.item()
                english_name = self.model.names[cls]
                class_name = self.names_map.get(english_name, english_name)
                defect_info += f"{i+1}. {class_name} (置信度: {conf:.2f})\n"

                # 绘制过滤后的检测框，文本使用英文名以避免 OpenCV 中文乱码
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{english_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_img, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 8, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if len(filtered_boxes) == 0:
                defect_info = "未检测到置信度大于0.50的缺陷"

            self.detection_done.emit(annotated_img, defect_info)
        except Exception as e:
            self.detection_done.emit(None, f"检测失败: {str(e)}")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        
        # 背景 - 使用指定图片
        self.bg_label = QtWidgets.QLabel(MainWindow)
        self.bg_label.setGeometry(0, 0, MainWindow.width(), MainWindow.height())
        self.bg_label.setScaledContents(True)
        bg_image_path = os.path.join(os.path.dirname(__file__), 'background.png')
        if os.path.exists(bg_image_path):
            bg_pixmap = QtGui.QPixmap(bg_image_path)
            self.bg_label.setPixmap(bg_pixmap)
        else:
            self.bg_label.setStyleSheet("background-color: #f0f0f0;")  # 备用浅灰色背景
        self.bg_label.lower()

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 标题
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(168, 60, 551, 71))
        self.label.setStyleSheet("font-size:42px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0);color:white;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        # 图像显示区
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(40, 188, 751, 501))
        self.image_label.setStyleSheet("background:rgba(255,255,255,1);")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("image_label")

        # 缺陷显示区
        self.defect_browser = QtWidgets.QTextBrowser(self.centralwidget)
        self.defect_browser.setGeometry(QtCore.QRect(820, 188, 400, 501))
        self.defect_browser.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF0000; background-color: #FFFFFF; border: 2px solid #000000;")
        self.defect_browser.setObjectName("defect_browser")
        self.defect_browser.setPlainText("缺陷检测结果")

        # 结果显示区
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(73, 746, 851, 174))
        self.textBrowser.setStyleSheet("background:rgba(0,0,0,0);")
        self.textBrowser.setObjectName("textBrowser")

        # 选择图像按钮
        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(1020, 750, 150, 40))
        self.select_button.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.select_button.setObjectName("select_button")

        # 检测按钮
        self.detect_button = QtWidgets.QPushButton(self.centralwidget)
        self.detect_button.setGeometry(QtCore.QRect(1020, 810, 150, 40))
        self.detect_button.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.detect_button.setObjectName("detect_button")

        # 退出按钮
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(1020, 870, 150, 40))
        self.exit_button.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.exit_button.setObjectName("exit_button")

        MainWindow.setCentralWidget(self.centralwidget)

        # 信号绑定
        self.select_button.clicked.connect(self.select_image)
        self.detect_button.clicked.connect(self.start_detection)
        self.exit_button.clicked.connect(self.exit_system)
           
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 加载模型
        self.model = None
        self.current_image_path = None
        # 中文类别映射
        self.class_names_map = {
            'crack': '裂纹',
            'spot': '斑点',
            'broken': '破损',
            'scratch': '划痕',
            'light-leakage': '漏光',
            'broken-membrane': '破膜',
            'blot': '污渍'
        }
        self.load_model()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手机屏幕缺陷检测系统"))
        self.label.setText(_translate("MainWindow", "手机屏幕缺陷检测系统"))
        self.image_label.setText(_translate("MainWindow", "请选择图像进行检测"))
        self.select_button.setText(_translate("MainWindow", "选择图像"))
        self.detect_button.setText(_translate("MainWindow", "开始检测"))
        self.exit_button.setText(_translate("MainWindow", "退出系统"))

    def load_model(self):
        try:
            # 使用训练好的模型
            model_path = "runs/train/ssgd_baseline_hyper/weights/best.pt"
            self.model = YOLO(model_path)
            self.printf("模型加载成功")
        except Exception as e:
            self.printf(f"模型加载失败: {str(e)}")

    def select_image(self):
        fname, _ = QFileDialog.getOpenFileName(None, "选择图像", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.current_image_path = fname
            self.printf(f"已选择图像: {fname}")
            # 显示原始图像
            img = cv2.imread(fname)
            if img is not None:
                self.show_image(img)

    def start_detection(self):
        if self.model is None:
            self.printf("模型未加载")
            return
        if self.current_image_path is None:
            self.printf("请先选择图像")
            return
        
        self.printf("开始检测...")
        self.detector_thread = DefectDetector(self.current_image_path, self.model, self.class_names_map)
        self.detector_thread.detection_done.connect(self.on_detection_done)
        self.detector_thread.start()

    def on_detection_done(self, annotated_img, info):
        if annotated_img is not None:
            self.show_image(annotated_img)
        self.defect_browser.setPlainText(info)
        self.printf("检测完成，结果已显示在缺陷区域。")

    def show_image(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _image = QImage(img2.data, img2.shape[1], img2.shape[0], 
                      img2.shape[1] * 3, QImage.Format_RGB888)
        scaled_img = _image.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled_img))

    def printf(self, text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def exit_system(self):
        os._exit(0)

    def on_resize(self, event):
        # 调整背景大小
        self.bg_label.setGeometry(0, 0, event.size().width(), event.size().height())
        # 如果需要，可以在这里调整其他控件的几何位置
        # 目前保留固定布局，避免在 resizeEvent 中重复创建控件

    def start_perspective_selection(self):
        self.selecting_points = []
        self.printf("请依次点击视频区域四个角点（顺时针，从左上开始）")

    def on_point_selected(self, pos):
        if hasattr(self, 'selecting_points'):
            label_w, label_h = self.label_2.width(), self.label_2.height()
            img_h, img_w = self.current_frame.shape[:2]
            x = int(pos[0] * img_w / label_w)
            y = int(pos[1] * img_h / label_h)
            self.selecting_points.append((x, y))
            self.printf(f"已选点: {len(self.selecting_points)}/4, 坐标: {x},{y}")
            if len(self.selecting_points) == 4:
                self.perspective_points = self.selecting_points.copy()
                self.printf("四点已选完，将用于透视变换。")
                # 启动识别线程，并传递四点
                self.thread_1.line_coords = None
                self.thread_1.perspective_points = self.perspective_points
                self.thread_1.start()

class ModelLoaderThread(QThread):
    model_loaded_signal = QtCore.pyqtSignal(object)
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
    def run(self):
        model = torch.hub.load('yolov5', 'custom', path=self.model_path,source='local')
        self.model_loaded_signal.emit(model)

class ClickableLabel(QtWidgets.QLabel):
    point_selected = QtCore.pyqtSignal(tuple)  # (x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            self.point_selected.emit((x, y))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())