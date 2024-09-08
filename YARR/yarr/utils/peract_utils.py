import numpy as np

def get_new_scene_bounds_based_on_crop(radius, target_object_pos):
    target_object_pos = np.round(target_object_pos, 2)
    new_scene_bounds = [target_object_pos[0] - radius,\
                        target_object_pos[1] - radius,\
                        target_object_pos[2] - radius,\
                        target_object_pos[0] + radius,\
                        target_object_pos[1] + radius,\
                        target_object_pos[2] + radius]
    return new_scene_bounds