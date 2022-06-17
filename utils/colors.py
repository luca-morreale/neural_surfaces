
import colorsys
import numpy as np


def sinebow(h):
    f = lambda x : np.sin(np.pi * x)**2
    return np.stack([f(3/6-h), f(5/6-h), f(7/6-h)], -1)

def hsv2rgb(hsv_list):
    return [ np.array(list(colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])) + [1.0])*255 for hsv in hsv_list ]
