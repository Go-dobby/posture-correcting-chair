import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import serial
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import *

PORT = 'COM6'
BaudRate = 9600
arduino = serial.Serial(PORT, BaudRate)

currentData_2 = [0, 0, 0, 0]  # 현재데이터 저장
stateCount_1 = 0  # 바른자세 카운트
stateCount_2 = 0  # 왼쪽
stateCount_3 = 0  # 오른쪽
stateCount_4 = 0  # 앞
stateCount_5 = 0  # 뒤

upCount_1 = 0
upCount_2 = 0

posture_1 = 0
posture_2 = 0
posture_3 = 0
posture_4 = 0
posture_5 = 0

Data = pd.read_csv(r'weight data file path')
Data = np.array(Data)

x = Data[:, 0:4]  # input 4
y = Data[:, 4:8]  # output 4

x_train = np.array(x)
y_train = np.array(y)

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  # 출력갯수, 입력갯수,
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128)

scores = model.evaluate(x_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


class Scope_1(object):

    def __init__(self,
                 ax_1, fn_1,
                 xmax=60, ymax=1000,
                 xstart=0, ystart=0,
                 xlabel='sec', ylabel='percentage'):
        self.xmax = xmax
        self.xstart = xstart
        self.ymax = ymax
        self.ystart = ystart

        self.ax_1 = ax_1
        self.ax_1.set_xlim((self.xstart, self.xmax))
        self.ax_1.set_ylim((self.ystart, self.ymax))
        self.ax_1.set_title('sensor1')
        self.ax_1.set_xlabel(xlabel, x=1)
        self.ax_1.set_ylabel(ylabel)

        self.x_1 = [0]
        self.y_1 = [0]
        self.value_1 = 0
        self.fn_1 = fn_1
        self.line_1, = ax_1.plot([], [])

        self.ti = time.time()
        print("초기화 완료")

    def update_1(self, i):
        tempo = time.time() - self.ti
        self.ti = time.time()

        self.value_1 = self.fn_1()
        self.y_1.append(self.value_1)
        self.x_1.append(tempo + self.x_1[-1])
        self.line_1.set_data(self.x_1, self.y_1)

        if self.x_1[-1] >= self.xstart + self.xmax:
            self.xstart = self.xstart + self.xmax / 2
            self.ax_1.set_xlim(self.xstart, self.xstart + self.xmax)

            self.ax_1.figure.canvas.draw()

        return (self.line_1,)


class Scope_2(object):

    def __init__(self,
                 ax_2, fn_2,
                 xmax=60, ymax=1000,
                 xstart=0, ystart=0,
                 xlabel='sec', ylabel='percentage'):
        self.xmax = xmax
        self.xstart = xstart
        self.ymax = ymax
        self.ystart = ystart

        self.ax_2 = ax_2
        self.ax_2.set_xlim((self.xstart, self.xmax))
        self.ax_2.set_ylim((self.ystart, self.ymax))
        self.ax_2.set_title('sensor2')
        self.ax_2.set_xlabel(xlabel, x=1)
        self.ax_2.set_ylabel(ylabel)

        self.x_2 = [0]
        self.y_2 = [0]
        self.value_2 = 0
        self.fn_2 = fn_2
        self.line_2, = ax_2.plot([], [])

        self.ti = time.time()
        print("초기화 완료")

    def update_2(self, i):
        tempo = time.time() - self.ti
        self.ti = time.time()

        self.value_2 = self.fn_2()
        self.y_2.append(self.value_2)
        self.x_2.append(tempo + self.x_2[-1])
        self.line_2.set_data(self.x_2, self.y_2)

        if self.x_2[-1] >= self.xstart + self.xmax:
            self.xstart = self.xstart + self.xmax / 2
            self.ax_2.set_xlim(self.xstart, self.xstart + self.xmax)

            self.ax_2.figure.canvas.draw()

        return (self.line_2,)


class Scope_3(object):

    def __init__(self,
                 ax_3, fn_3,
                 xmax=60, ymax=1000,
                 xstart=0, ystart=0,
                 xlabel='sec', ylabel='percentage'):
        self.xmax = xmax
        self.xstart = xstart
        self.ymax = ymax
        self.ystart = ystart

        self.ax_3 = ax_3
        self.ax_3.set_xlim((self.xstart, self.xmax))
        self.ax_3.set_ylim((self.ystart, self.ymax))
        self.ax_3.set_title('sensor3')
        self.ax_3.set_xlabel(xlabel, x=1)
        self.ax_3.set_ylabel(ylabel)

        self.x_3 = [0]
        self.y_3 = [0]
        self.value_3 = 0
        self.fn_3 = fn_3
        self.line_3, = ax_3.plot([], [])

        self.ti = time.time()
        print("초기화 완료")

    def update_3(self, i):
        tempo = time.time() - self.ti
        self.ti = time.time()

        self.value_3 = self.fn_3()
        self.y_3.append(self.value_3)
        self.x_3.append(tempo + self.x_3[-1])
        self.line_3.set_data(self.x_3, self.y_3)

        if self.x_3[-1] >= self.xstart + self.xmax:
            self.xstart = self.xstart + self.xmax / 2
            self.ax_3.set_xlim(self.xstart, self.xstart + self.xmax)

            self.ax_3.figure.canvas.draw()

        return (self.line_3,)


class Scope_4(object):

    def __init__(self,
                 ax_4, fn_4,
                 xmax=60, ymax=1000,
                 xstart=0, ystart=0,
                 xlabel='sec', ylabel='percentage'):
        self.xmax = xmax
        self.xstart = xstart
        self.ymax = ymax
        self.ystart = ystart

        self.ax_4 = ax_4
        self.ax_4.set_xlim((self.xstart, self.xmax))
        self.ax_4.set_ylim((self.ystart, self.ymax))
        self.ax_4.set_title('sensor4')
        self.ax_4.set_xlabel(xlabel, x=1)
        self.ax_4.set_ylabel(ylabel)

        self.x_4 = [0]
        self.y_4 = [0]
        self.value_4 = 0
        self.fn_4 = fn_4
        self.line_4, = ax_4.plot([], [])

        self.ti = time.time()
        print("초기화 완료")

    def update_4(self, i):
        tempo = time.time() - self.ti
        self.ti = time.time()

        self.value_4 = self.fn_4()
        self.y_4.append(self.value_4)
        self.x_4.append(tempo + self.x_4[-1])
        self.line_4.set_data(self.x_4, self.y_4)

        if self.x_4[-1] >= self.xstart + self.xmax:
            self.xstart = self.xstart + self.xmax / 2
            self.ax_4.set_xlim(self.xstart, self.xstart + self.xmax)

            self.ax_4.figure.canvas.draw()

        return (self.line_4,)


class Scope_5(object):

    def __init__(self,
                 ax_5, fnc_1, fnc_2, fnc_3, fnc_4, fnc_5,
                 xmax=60, ymax=100,
                 xstart=0, ystart=0,
                 xlabel='sec', ylabel='count'):
        self.xmax = xmax
        self.xstart = xstart
        self.ymax = ymax
        self.ystart = ystart

        self.ax_5 = ax_5
        self.ax_5.set_xlim((self.xstart, self.xmax))
        self.ax_5.set_ylim((self.ystart, self.ymax))
        self.ax_5.set_title('counter')
        self.ax_5.set_xlabel(xlabel, x=1)
        self.ax_5.set_ylabel(ylabel)

        self.cx_1 = [0]
        self.cx_2 = [0]
        self.cx_3 = [0]
        self.cx_4 = [0]
        self.cx_5 = [0]

        self.cy_1 = [0]
        self.cy_2 = [0]
        self.cy_3 = [0]
        self.cy_4 = [0]
        self.cy_5 = [0]

        self.cvalue_1 = 0
        self.cvalue_2 = 0
        self.cvalue_3 = 0
        self.cvalue_4 = 0
        self.cvalue_5 = 0

        self.fnc_1 = fnc_1
        self.fnc_2 = fnc_2
        self.fnc_3 = fnc_3
        self.fnc_4 = fnc_4
        self.fnc_5 = fnc_5

        self.linec_1, = ax_5.plot([], [], color='red', label='steady')
        self.linec_2, = ax_5.plot([], [], color='green', label='left')
        self.linec_3, = ax_5.plot([], [], color='blue', label='right')
        self.linec_4, = ax_5.plot([], [], color='black', label='forward')
        self.linec_5, = ax_5.plot([], [], color='purple', label='back')

        self.ti = time.time()
        print("초기화 완료")

    def update_5(self, i):

        tempo = time.time() - self.ti
        self.ti = time.time()

        self.cvalue_1 = self.fnc_1()
        self.cvalue_2 = self.fnc_2()
        self.cvalue_3 = self.fnc_3()
        self.cvalue_4 = self.fnc_4()
        self.cvalue_5 = self.fnc_5()

        self.cy_1.append(self.cvalue_1)
        self.cy_2.append(self.cvalue_2)
        self.cy_3.append(self.cvalue_3)
        self.cy_4.append(self.cvalue_4)
        self.cy_5.append(self.cvalue_5)

        self.cx_1.append(tempo + self.cx_1[-1])
        self.cx_2.append(tempo + self.cx_2[-1])
        self.cx_3.append(tempo + self.cx_3[-1])
        self.cx_4.append(tempo + self.cx_4[-1])
        self.cx_5.append(tempo + self.cx_5[-1])

        self.linec_1.set_data(self.cx_1, self.cy_1)
        self.linec_2.set_data(self.cx_2, self.cy_2)
        self.linec_3.set_data(self.cx_3, self.cy_3)
        self.linec_4.set_data(self.cx_4, self.cy_4)
        self.linec_5.set_data(self.cx_5, self.cy_5)

        global posture_1
        global posture_2
        global posture_3
        global posture_4
        global posture_5

        global upCount_1
        global upCount_2

        global stateCount_1
        global stateCount_2
        global stateCount_3
        global stateCount_4
        global stateCount_5

        if (self.cx_1[-1] >= int((self.xstart + self.xmax) / 2 - 1)) and upCount_1 == 0:
            posture_1 = stateCount_1
            posture_2 = stateCount_2
            posture_3 = stateCount_3
            posture_4 = stateCount_4
            posture_5 = stateCount_5

        if (self.cx_1[-1] >= (self.xstart + self.xmax) / 2) and upCount_1 == 0:  # 첫 30초

            stateCount_1 = 0
            stateCount_2 = 0
            stateCount_3 = 0
            stateCount_4 = 0
            stateCount_5 = 0
            upCount_1 = 1
            upCount_2 = 1
            if posture_1 > posture_2 and posture_1 > posture_3 and \
                    posture_1 > posture_4 and posture_1 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "바른자세 \n30초 뒤에 자동으로 닫힙니다.", "바른자세",
                                                 icon=QMessageBox.NoIcon)

            elif posture_2 > posture_1 and posture_2 > posture_3 and \
                    posture_2 > posture_4 and posture_2 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "왼쪽 \n30초 뒤에 자동으로 닫힙니다.", "왼쪽",
                                                 icon=QMessageBox.Critical)

            elif posture_3 > posture_1 and posture_3 > posture_2 and \
                    posture_3 > posture_4 and posture_3 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "오른쪽 \n30초 뒤에 자동으로 닫힙니다.", "오른쪽",
                                                 icon=QMessageBox.Critical)

            elif posture_4 > posture_1 and posture_4 > posture_2 and \
                    posture_4 > posture_3 and posture_4 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "앞 \n30초 뒤에 자동으로 닫힙니다.", "앞",
                                                 icon=QMessageBox.Critical)

            elif posture_5 > posture_1 and posture_5 > posture_2 and \
                    posture_5 > posture_3 and posture_5 > posture_4 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "뒤 \n30초 뒤에 자동으로 닫힙니다.", "뒤",
                                                 icon=QMessageBox.Critical)

        if self.cx_1[-1] >= (int(self.xstart + self.xmax) - 1):
            posture_1 = stateCount_1
            posture_2 = stateCount_2
            posture_3 = stateCount_3
            posture_4 = stateCount_4
            posture_5 = stateCount_5

        if self.cx_1[-1] >= self.xstart + self.xmax:
            stateCount_1 = 0
            stateCount_2 = 0
            stateCount_3 = 0
            stateCount_4 = 0
            stateCount_5 = 0
            upCount_2 = 1
            if posture_1 > posture_2 and posture_1 > posture_3 and \
                    posture_1 > posture_4 and posture_1 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "바른자세 \n30초 뒤에 자동으로 닫힙니다.", "바른자세",
                                                 icon=QMessageBox.NoIcon)

            elif posture_2 > posture_1 and posture_2 > posture_3 and \
                    posture_2 > posture_4 and posture_2 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "왼쪽 \n30초 뒤에 자동으로 닫힙니다.", "왼쪽",
                                                 icon=QMessageBox.Critical)

            elif posture_3 > posture_1 and posture_3 > posture_2 and \
                    posture_3 > posture_4 and posture_3 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "오른쪽 \n30초 뒤에 자동으로 닫힙니다.", "오른쪽",
                                                 icon=QMessageBox.Critical)

            elif posture_4 > posture_1 and posture_4 > posture_2 and \
                    posture_4 > posture_3 and posture_4 > posture_5 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "앞 \n30초 뒤에 자동으로 닫힙니다.", "앞",
                                                 icon=QMessageBox.Critical)

            elif posture_5 > posture_1 and posture_5 > posture_2 and \
                    posture_5 > posture_3 and posture_5 > posture_4 and upCount_2 == 1:

                upCount_2 = 0

                CustomMessageBox.showWithTimeout(28, "뒤 \n30초 뒤에 자동으로 닫힙니다.", "뒤",
                                                 icon=QMessageBox.Critical)

            self.xstart = self.xstart + self.xmax / 2
            self.ax_5.set_xlim(self.xstart, self.xstart + self.xmax)
            self.ax_5.figure.canvas.draw()

        return (self.linec_1, self.linec_2, self.linec_3, self.linec_4, self.linec_5,)





class CustomMessageBox(QMessageBox):

    def __init__(self, *__args):
        QMessageBox.__init__(self)
        self.left = 1150
        self.top = 750
        self.width = 1000
        self.height = 1000
        self.timeout = 0
        self.autoclose = False
        self.currentTime = 0
        self.setGeometry(self.left, self.top, self.width, self.height)

    def showEvent(self, QShowEvent):
        self.currentTime = 0
        if self.autoclose:
            self.startTimer(1000)

    def timerEvent(self, *args, **kwargs):
        self.currentTime += 1
        if self.currentTime >= self.timeout:
            self.done(0)

    @staticmethod
    def showWithTimeout(timeoutSeconds, message, title, icon=QMessageBox.Information, buttons=QMessageBox.Ok):
        w = CustomMessageBox()
        w.autoclose = True
        w.timeout = timeoutSeconds
        w.setText(message)
        w.setWindowTitle(title)
        w.setIcon(icon)
        w.setStandardButtons(buttons)
        w.exec_()


fig = plt.figure(figsize=(16, 10))
ax_1 = plt.subplot(231)
ax_1.grid(True)

ax_2 = plt.subplot(232)
ax_2.grid(True)

ax_3 = plt.subplot(233)
ax_3.grid(True)

ax_4 = plt.subplot(234)
ax_4.grid(True)

ax_5 = plt.subplot(235)
ax_5.grid(True)


def insert_1():
    currentData = []
    data = (arduino.readline()).decode()
    data = str(data)

    while data[0] != 'A':
        data = (arduino.readline()).decode()
        data = str(data)

    print(data[:-2] + ' #1')
    currentData.append(data[1:-2])
    currentData = list(map(int, currentData))
    sensorValue1 = int(sum(currentData))
    value_1 = sensorValue1

    global currentData_2
    del currentData_2[0:1]
    currentData_2.insert(0, value_1)

    return value_1


def insert_2():
    currentData = []
    data = (arduino.readline()).decode()
    data = str(data)

    while data[0] != 'B':
        data = (arduino.readline()).decode()
        data = str(data)

    print(data[:-2] + ' #2')
    currentData.append(data[1:-2])
    currentData = list(map(int, currentData))
    sensorValue2 = int(sum(currentData))
    value_2 = sensorValue2

    global currentData_2
    del currentData_2[1:2]
    currentData_2.insert(1, value_2)

    return value_2


def insert_3():
    currentData = []
    data = (arduino.readline()).decode()
    data = str(data)

    while data[0] != 'C':
        data = (arduino.readline()).decode()
        data = str(data)

    print(data[:-2] + ' #3')
    currentData.append(data[1:-2])
    currentData = list(map(int, currentData))
    sensorValue3 = int(sum(currentData))
    value_3 = sensorValue3

    global currentData_2
    del currentData_2[2:3]
    currentData_2.insert(2, value_3)
    print(currentData_2)
    return value_3


def insert_4():
    currentData = []
    data = (arduino.readline()).decode()
    data = str(data)

    while data[0] != 'D':
        data = (arduino.readline()).decode()
        data = str(data)

    print(data[:-2] + ' #4')
    currentData.append(data[1:-2])
    currentData = list(map(int, currentData))
    sensorValue4 = int(sum(currentData))
    value_4 = sensorValue4

    global currentData_2
    global Data
    global stateCount_1
    global stateCount_2
    global stateCount_3
    global stateCount_4
    global stateCount_5

    del currentData_2[3:4]
    currentData_2.insert(3, value_4)
    print(currentData_2)
    Data = []

    for i in range(0, len(currentData_2)):
        Data.append(currentData_2[i])

    Data = np.array([Data])
    y_predict = model.predict(Data)
    print(y_predict)

    position_1 = y_predict[:, 0]  # 정면
    position_2 = y_predict[:, 1]  # 왼쪽
    position_3 = y_predict[:, 2]  # 오른쪽
    position_4 = y_predict[:, 3]  # 일어남
    dataF = []
    dataF = (arduino.readline()).decode()

    while (dataF[0] != 'E') and (dataF[0] != 'F') and (dataF[0] != 'G') and (dataF[0] != 'I') and (dataF[0] != 'H'):
        dataF = (arduino.readline()).decode()
        dataF = str(dataF)
        if dataF[0] == 'E':
            break
        elif dataF[0] == 'F':
            break
        elif dataF[0] == 'G':
            break
        elif dataF[0] == 'H':
            break
        elif dataF[0] == 'I':
            break

    if np.all(position_1 > position_2 and position_1 > position_3 and position_1 > position_4):


        if dataF[0] == 'E':
            stateCount_1 += 1
            print("바른자세")
        elif dataF[0] == 'F' or dataF[0] == 'I':
            stateCount_4 += 1
            print("앞")
        elif dataF[0] == 'G':
            stateCount_5 += 1
            print("뒤")
    elif dataF[0] == 'F' or dataF[0] == 'I':
        stateCount_4 += 1
        print("앞")
    elif dataF[0] == 'G':
        stateCount_5 += 1
        print("뒤")
    elif np.all(position_2 > position_1 and position_2 > position_3 and position_2 > position_4):
        print("왼쪽")
        stateCount_2 += 1
        print('statecount2 = {0}'.format(stateCount_2))

    elif np.all(position_3 > position_1 and position_3 > position_2 and position_3 > position_4 ):
        print("오른쪽")
        stateCount_3 += 1
        print('statecount3 = {0}'.format(stateCount_3))

    elif np.all(position_4 > position_1 and position_4 > position_2 and position_4 > position_3 ):
        print("일어남")

    else:
        print("측정불가")

    return value_4


def insert_5():
    return 0


def counter_1():
    global stateCount_1
    return stateCount_1


def counter_2():
    global stateCount_2
    return stateCount_2


def counter_3():
    global stateCount_3
    return stateCount_3


def counter_4():
    global stateCount_4
    return stateCount_4


def counter_5():
    global stateCount_5
    return stateCount_5


scope_1 = Scope_1(ax_1, insert_1)
scope_2 = Scope_2(ax_2, insert_2)
scope_3 = Scope_3(ax_3, insert_3)
scope_4 = Scope_4(ax_4, insert_4)
scope_5 = Scope_5(ax_5, counter_1, counter_2, counter_3, counter_4, counter_5)


ani_1 = animation.FuncAnimation(fig, scope_1.update_1, interval=50, blit=True)
ani_2 = animation.FuncAnimation(fig, scope_2.update_2, interval=50, blit=True)
ani_3 = animation.FuncAnimation(fig, scope_3.update_3, interval=50, blit=True)
ani_4 = animation.FuncAnimation(fig, scope_4.update_4, interval=50, blit=True)
ani_5 = animation.FuncAnimation(fig, scope_5.update_5, interval=50, blit=True)


ax_5.legend()
plt.show()