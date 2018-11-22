import math
from tkinter import *
import random
#from Road import TRoad


class TCar:

    def __init__(self, x, y, v, angle, canvas: Canvas):
        self.x = x
        self.y = y
        self.v = v
        self.angle = angle
        self.isLive = True
        self.sqr = canvas.create_rectangle(30, 150, 5, 155, outline="#f50", fill="#f50")

    def move(self, step):
        """Изменение координат машинки"""
        if self.isLive:
            self.x += self.v * math.cos(self.angle)*step
            self.y -= self.v * math.sin(self.angle)*step

    def turn_left(self, phi=math.pi/15.0):
        """Поворот вектора скорости против часовой стрелки на угол phi"""
        self.angle += phi
        if self.angle > 2 * math.pi:
            self.angle -= 2 * math.pi

    def turn_right(self, phi=math.pi/13.0):
        """Поворот вектора скорости по часовой стрелки на угол phi"""
        self.angle -= phi
        if self.angle < 0:
            self.angle += 2 * math.pi
