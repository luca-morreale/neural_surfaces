
from math import sin, pi


derekBlue = (144.0/255, 210.0/255, 236.0/255, 1)
coralRed = (250.0/255, 114.0/255, 104.0/255, 1)
iglGreen = (153.0/255, 203.0/255, 67.0/255, 1)
caltechOrange = (255.0/255, 108.0/255, 12.0/255, 1)
royalBlue = (0/255, 35/255, 102/255, 1)
royalYellow = (250.0/255,218.0/255,94.0/255, 1)
white = (1,1,1,1)
black = (0,0,0,1)
red = (250.0/255, 0.0, 0.0, 1)

# color palette for color blindness (source: http://mkweb.bcgsc.ca/colorblind)
cb_black = (0/255.0, 0/255.0, 236/255.0, 1)
cb_orange = (230/255.0, 159/255.0, 0/255.0, 1)
cb_skyBlue = (86/255.0, 180/255.0, 233/255.0, 1)
cb_green = (0/255.0, 158/255.0, 115/255.0, 1)
cb_yellow = (240/255.0, 228/255.0, 66/255.0, 1)
cb_blue = (0/255.0, 114/255.0, 178/255.0, 1)
cb_vermillion = (213/255.0, 94/255.0, 0/255.0, 1)
cb_purple = (204/255.0, 121/255.0, 167/255.0, 1)


class discreteColor(object):
    def __init__(self, brightness = 0, pos1 = 0, pos2 = 0):
        self.brightness = brightness
        self.rampElement1_pos = pos1
        self.rampElement2_pos = pos2

class colorObj(object):
    def __init__(self, RGBA = derekBlue, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # birghtness
        self.C = C # contrast

def sinebow_color(h):
    f = lambda x : sin(pi * x)**2
    return (f(3/6-h), f(5/6-h), f(7/6-h), 1)
