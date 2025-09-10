import bpy
import numpy as np

def get_obj_position(name):
    obj = bpy.data.objects.get(name)
    return np.array(obj.location) if obj else np.zeros(3)

def get_engines_location():
    pos_L = get_obj_position("Engin_L")
    pos_R = get_obj_position("Engin_R")
    return pos_L, pos_R