from Car import TCar
import math
from tkinter import *
from PIL import Image
import numpy as np
from scipy import ndimage

class TRoad:

    def __init__(self, n, RoadFunction, weight):
        """в конструкторе создаем массив карты
        n - размер матрицы среза, RoadFunction - математическое выражение описывающее трассу, weight - ширина трассы"""
        self.PictureArr = []
        self.n = n
        self.b = int((n-1)/2)
        self.RoadFunction = RoadFunction
        self.weight = weight
        for i in range(628):
            self.PictureArr.append([])
            for j in range(300):
                if j-self.weight < self.RoadFunction(i) < j+self.weight:
                    self.PictureArr[i].append(0)
                else:
                    self.PictureArr[i].append(255)

    def paint(self, canvas: Canvas):
        """ процедуры рисования массива карты на canvas """
        for i in range(628):
            for j in range(300):
                if self.PictureArr[i][j] < 255:
                    canvas.create_rectangle(i, j, i, j, outline="#ffffff", fill="#ffffff")

    def check_is_live(self, car: TCar):
        """проверка жизни машинки"""
        if self.PictureArr[round(car.x)][round(car.y)] == 255:
            car.isLive = False

    def slice_train(self, car: TCar) -> list:
        """срезы массива карты для получения изображения видимости машинки"""
        firstSliceArr = []
        for j in range(self.n * 2):
            firstSliceArr.append([])
            for i in range(self.n * 2):
                firstSliceArr[-1].append(self.PictureArr[round(car.x) + 2 * self.b - i][round(car.y) - 2 * self.b + j])

        res = []
        for i in range(self.n * 2):
            res.append([])
            for j in range(self.n * 2):
                res[i].append(0)

        for i in range(self.n * 2):
            for j in range(self.n * 2):
                phi = math.atan2(j - self.n, i - self.n)
                ro = ((i - self.n) ** 2 + (j - self.n) ** 2) ** 0.5
                phi += car.angle
                x = self.n + ro * math.cos(phi)
                y = self.n + ro * math.sin(phi)
                if 1 <= round(x) < 2 * self.n - 1 and 1 <= round(y) < 2 * self.n - 1:
                    res[round(x) - 1][round(y)] = firstSliceArr[i][j]
                    res[round(x) - 1][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x) - 1][round(y) + 1] = firstSliceArr[i][j]
                    res[round(x)][round(y)] = firstSliceArr[i][j]
                    res[round(x)][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x)][round(y) + 1] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y)] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y) + 1] = firstSliceArr[i][j]

        finalSliceArr = []
        for i in range(self.n):
            finalSliceArr.append([])
            for j in range(self.n):
                finalSliceArr[-1].append(res[self.b + i][self.b + j]*0.99/255+0.01)

        return finalSliceArr

    def slice_see(self, car: TCar) -> list:
        """Получение среза машинки"""
        firstSliceArr = []
        for j in range(self.n*2):
            firstSliceArr.append([])
            for i in range(self.n*2):
                firstSliceArr[-1].append(self.PictureArr[round(car.x)+2*self.b-i][round(car.y)-2*self.b+j])

        res = []
        for i in range(self.n*2):
            res.append([])
            for j in range(self.n*2):
                res[i].append(0)

        for i in range(self.n*2):
            for j in range(self.n*2):
                phi = math.atan2(j - self.n, i - self.n)
                ro = ((i - self.n) ** 2 + (j - self.n) ** 2) ** 0.5
                phi += car.angle
                x = self.n + ro * math.cos(phi)
                y = self.n + ro * math.sin(phi)
                if 1 <= round(x) < 2*self.n - 1 and 1 <= round(y) < 2*self.n - 1:
                    res[round(x) - 1][round(y)] = firstSliceArr[i][j]
                    res[round(x) - 1][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x) - 1][round(y) + 1] = firstSliceArr[i][j]
                    res[round(x)][round(y)] = firstSliceArr[i][j]
                    res[round(x)][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x)][round(y) + 1] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y)] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y) - 1] = firstSliceArr[i][j]
                    res[round(x) + 1][round(y) + 1] = firstSliceArr[i][j]

        finalSliceArr = []
        for i in range(self.n):
            finalSliceArr.append([])
            for j in range(self.n):
                finalSliceArr[-1].append(res[self.b+i][self.b+j])

        return finalSliceArr
