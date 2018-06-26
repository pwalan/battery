# coding:utf-8


class Particle(object):
    num = 0

    def __init__(self, position, velocity, value, best_postion, best_value):
        self.position = position
        self.velocity = velocity
        self.value = value
        self.best_postion = best_postion
        self.best_value = best_value
