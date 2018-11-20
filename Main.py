import numpy
from Car import TCar
from Road import TRoad
from NeuralNetwork import NeuralNetwork
from tkinter import *
import math
import random

dt = 0.7

def htmlcolor(r, g, b):
    def _chkarg(a):
        if isinstance(a, int): # clamp to range 0--255
            if a < 0:
                a = 0
            elif a > 255:
                a = 255
        elif isinstance(a, float): # clamp to range 0.0--1.0 and convert to integer 0--255
            if a < 0.0:
                a = 0
            elif a > 1.0:
                a = 255
            else:
                a = int(round(a*255))
        else:
            raise ValueError('Arguments must be integers or floats.')
        return a
    r = _chkarg(r)
    g = _chkarg(g)
    b = _chkarg(b)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def respawn(car: TCar, road: TRoad):
    """Перемещение машинки в заданные координаты"""
    car.isLive = False
    while not car.isLive:
        car.x = random.randint(100, 400)
        for i in range(100, 295):
            if road.PictureArr[car.x][i] == 0 and road.PictureArr[car.x][i-5] == 0 \
                    and road.PictureArr[car.x][i+5] == 0 and road.PictureArr[car.x-5][i] == 0 \
                    and road.PictureArr[car.x+5][i] == 0:
                car.y = i
                car.isLive = True
        car.angle = 0


# Вывод экрана с областью видимости машинки
def draweyecar(car: TCar, canvas: Canvas, road: TRoad):
    Arr = road.slice_see(car)
    for i in range(road.n):
        for j in range(road.n):
            if Arr[i][j] < 255:
                canvas.create_rectangle(50+i, 400+j, 50+i, 400+j, outline="#ffffff", fill="#ffffff")
            else:
                canvas.create_rectangle(50 + i, 400 + j, 50 + i, 400 + j)
    canvas.create_rectangle(50+road.b, 400+road.b, 50+road.b, 400+road.b, outline="#f50", fill="#f50")

# Вывод экранов показывающих чему обучилась машинка
def checkstudy(neural: NeuralNetwork, canvas: Canvas):
    label123 = 0
    targets = numpy.zeros(neural.onodes) + 0.01
    targets[label123] = 0.99
    inputs = neural.backquery(targets)
    Arr = []
    for i in range(neural.inodes):
        Arr.append(255-(inputs[i][0]-0.01)*255/0.99)
    for i in range(road.n):
        for j in range(road.n):
           # color = htmlcolor(int(round(Arr[i*51+j])),int(round(Arr[i*51+j])), int(round(Arr[i*51+j])))
            canvas.create_rectangle(150 + i, 400 + j, 150 + i, 400 + j, outline=htmlcolor(int(round(Arr[i*road.n+j])),int(round(Arr[i*road.n+j])), int(round(Arr[i*road.n+j]))), fill=htmlcolor(int(round(Arr[i*road.n+j])),int(round(Arr[i*road.n+j])), int(round(Arr[i*road.n+j]))))

    label123 = 1
    targets = numpy.zeros(neural.onodes) + 0.01
    targets[label123] = 0.99
    inputs = neural.backquery(targets)
    Arr = []
    for i in range(neural.inodes):
        Arr.append(255-(inputs[i][0] - 0.01) * 255 / 0.99)
    for i in range(road.n):
        for j in range(road.n):
            canvas.create_rectangle(250 + i, 400 + j, 250 + i, 400 + j, outline=htmlcolor(int(round(Arr[i*road.n+j])),int(round(Arr[i*road.n+j])), int(round(Arr[i*road.n+j]))), fill=htmlcolor(int(round(Arr[i*road.n+j])),int(round(Arr[i*road.n+j])), int(round(Arr[i*road.n+j]))))



# Движение, перерисовка и обучение машинки
def update_draws_car(car: TCar, canvas: Canvas, road: TRoad):
    car.move(dt)
    canvas.coords(car.sqr, car.x-2, car.y-2, car.x+2, car.y+2)
    road.check_is_live(car)
    machine_learning(car, road)

# Движение, перерисовка и принятие решения о повороте
def update_draws_car_machine(car: TCar, canvas: Canvas, road: TRoad):
    car.move(dt)
    canvas.coords(car.sqr, car.x-2, car.y-2, car.x+2, car.y+2)
    road.check_is_live(car)
    Arr = road.slice_train(car)
    SliceArr = []
    for i in range(road.n):
        for j in range(road.n):
            SliceArr.append(Arr[i][j])
    outputs = neural.query(SliceArr)
    label = numpy.argmax(outputs)
    if label == 0:
        car.turn_left()
    if label == 1:
        car.turn_right()

# Обучение машинки в случае смерти
def machine_learning(car: TCar, road: TRoad):
    Arr = road.slice_train(car)
 #   outputs = n.query(Arr)
  #  label = numpy.argmax(outputs)
    label = random.randint(0, 1)
    if label == 0:
        car.turn_left()
    if label == 1:
        car.turn_right()
    study = label
    if not car.isLive:
        #study = train_with_road_function(car, road)
        study = train_with_slices(Arr)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(study)] = 0.99
        SliceArr = []
        for i in range(road.n):
            for j in range(road.n):
                SliceArr.append(Arr[i][j])
        neural.train(SliceArr, targets)


# обучение с использованием мат. выражения трассы
def train_with_road_function(car: TCar, road: TRoad) ->int:
    if road.RoadFunction(car.x) > car.y:
        if car.angle > 3 * math.pi / 2 or car.angle < math.pi / 2:
            study = 1
        else:
            study = 0

    else:
        if car.angle > 3 * math.pi / 2 or car.angle < math.pi / 2:
            study = 0
        else:
            study = 1
    return study


# обучение с использованием срезов
def train_with_slices(SliceArr: list) ->int:
    size = SliceArr.__len__()
    LeftSum = 0
    for i in range(int(3 * size / 8), int(size / 2)):
        for j in range(int(3 * size / 8), int(size / 2)):
            LeftSum += SliceArr[i][j]

    RightSum = 0
    for i in range(int(size / 2), int(5 * size / 8)):
        for j in range(int(3 * size / 8), int(size / 2)):
            RightSum += SliceArr[i][j]

    if LeftSum < RightSum:
        return 0
    else:
        return 1


root = Tk()
canvas = Canvas(root, width=628, height=600)
canvas.pack()
canvas.configure(background='black')

n = 61
RoadFunction = lambda x: 30 * math.sin(x / 30) + 150
RoadFunction2 = lambda x: ((100 ** 2 - x ** 2)**2)**0.25 + 150
weight = 15
road = TRoad(n, RoadFunction, weight)
road.paint(canvas)

car = TCar(85, 150, 10, 0, canvas)
draweyecar(car, canvas, road)
canvas.update()
#car1 = TCar(190, 150, 10, 0, canvas)
input_nodes = road.n*road.n
hidden_nodes = 200
output_nodes = 2
learning_rate = 0.1
neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Цикл обучения
for i in range(50):
    respawn(car, road)
    while car.isLive:
        update_draws_car(car, canvas, road)
       # draweyecar(car, canvas, road)
       # checkstudy(neural, canvas)
        if car.x > 540:
            respawn(car, road)
       # canvas.update()
    print(i)


# for i in range(30):
#     x = random.randint(85, 400)
#     y = 8*math.sin(x/30)+150
#     angle = random.random()*2*math.pi
#     car.respawn(x, y, angle)
# #    car1.respawn(190, 150, math.pi)
#     while car.isLive: #or car1.isLive:
#         updatedrawscar(car, canvas, road)
#        # updatedrawscar(car1, canvas, road)
#         draweyecar(car, canvas, road)
#         checkstudy(n, canvas)
#         print(car.angle / (2 * math.pi) * 360)
#         canvas.update()
#     print(i)

mid = 0
# Цикл езды после обучения, на расчитанных весах
for i in range(10):
    respawn(car, road)
    while car.isLive:
        update_draws_car_machine(car, canvas, road)
        #draweyecar(car, canvas, road)
        #checkstudy(neural, canvas)
        canvas.update()
        print(car.angle*180/math.pi)
        if car.x > 540:
            respawn(car, road)
        mid += 1
    print(i)
print("mid =", mid/100)

canvas.mainloop()
