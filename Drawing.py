import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QFormLayout
from PyQt5.QtGui import QPainter, QPixmap, QPen
from PyQt5.QtCore import Qt, QPoint
import ModelUtils
import knn
import GetNumber


class Drawing(QWidget):
    def __init__(self, parent=None):
        super(Drawing, self).__init__(parent)
        self.setWindowTitle("title")
        self.pix = QPixmap()
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.clear = True
        self.initUi()
        self.Data = GetNumber.Data()
        self.knn_clf = None


    def initUi(self):

        self.resize(600, 600)
        # 画布大小为600*600，背景为白色
        self.pix = QPixmap(600, 600)
        self.pix.fill(Qt.white)


    def paintEvent(self, event):
        # pp = QPainter(self.pix)
        # # 根据鼠标指针前后两个位置绘制直线
        # pp.drawLine(self.lastPoint, self.endPoint,200)
        # # 让前一个坐标值等于后一个坐标值，
        # # 这样就能实现画出连续的线
        # self.lastPoint = self.endPoint
        # painter = QPainter(self)
        # painter.drawPixmap(0, 0, self.pix)

        pp = QPainter(self.pix)
        if self.clear == False:
            pen = QPen(Qt.black, 30, Qt.SolidLine)
            pp.setPen(pen)
            pp.drawLine(self.lastPoint, self.endPoint)
            self.lastPoint = self.endPoint
        self.clear = False
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        buttons = event.buttons()
        if buttons and Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def keyPressEvent(self, event):
        k = event.key()

        # load model
        if k == Qt.Key_L:
            self.knn_clf = knn.KNN(None, None)
            self.knn_clf.clf = ModelUtils.load_model('./model/model1.m')
            print("加载模型成功")
            return

        # save model
        if k == Qt.Key_S:
            if self.knn_clf.clf == None:
                print("模型为None，不能保存")
                return
            ModelUtils.save_model(self.knn_clf.clf, './model/model1.m')
            print("已保存模型")
            return


        # fit
        if k == Qt.Key_T:
            self.knn_clf = knn.KNN(self.Data.X, self.Data.y, is_pca = False)
            self.knn_clf.fitWithoutPca()

            # self.knn_clf = knn.KNN()
            # self.knn_clf.fitWithoutPca(self.Data.X, self.Data.y)
            return

        # 数据转为训练集并，显示X和y的shape
        if k == Qt.Key_W:
            self.Data.get_X_y()
            print(self.Data.X.shape)
            print(self.Data.y.shape)
            return

        # 显示数据个数
        if k == Qt.Key_E:
            print(len(self.Data.sub_imgs))
            print(len(self.Data.target))
            return

        # 清屏
        if k == Qt.Key_Q:
            self.clearScreen()
            self.update()
            return

        # 构造数据
        if k == Qt.Key_Space:
            if self.knn_clf != None:
                print("模型已经存在，不需要构造")
                return
            x = self.pos().x()
            y = self.pos().y()
            x = x + 10
            y = y + 50

            h, w = self.height() - 20, self.width()
            screen = QApplication.primaryScreen()

            pix = screen.grabWindow(0, x, y, w, h)
            pix.save("draw.jpg")
            # numbers = GetNumber.read_img()
            self.Data.getNumber()
            # self.show_number(numbers)
            return

        # 识别
        if k == Qt.Key_F:
            x = self.pos().x()
            y = self.pos().y()
            x = x + 10
            y = y + 50

            h, w = self.height() - 20, self.width()
            screen = QApplication.primaryScreen()

            pix = screen.grabWindow(0, x, y, w, h)
            pix.save("draw.jpg")
            numbers = GetNumber.read_img(self.knn_clf.clf)
            self.show_number(numbers)

    def show_number(self, numbers):
        print("检测结果从左往右依次为：" + str(numbers))

    def clearScreen(self):
        self.clear = True
        self.pix.fill(Qt.white)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Drawing()
    form.show()
    sys.exit(app.exec_())
    
# 空格：构造数据
# w:数据转为训练集
# L：加载模型
# S: 保存模型
# T：fit
# Q: 清屏
# F: 识别
