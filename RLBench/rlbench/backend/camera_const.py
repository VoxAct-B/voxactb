from pyquaternion import Quaternion

# 0527 left arm calibration: Errors 0.0223198 m, 0.00828765 rad
# left_translation = [0.14664, -0.38822, 0.37781] # x y z
# left_rotation_quat = [-0.75076, 0.17819, -0.19165, 0.60652] # (x, y, z, w)

# 0611 left arm calibration: Errors 0.0166798 m, 0.00553376 rad
left_translation = [0.15156, -0.41381, 0.39276] # x y z
left_rotation_quat = [-0.74793, 0.22418, -0.18728, 0.59605] # (x, y, z, w)

# (w, x, y, z)
left_rotation_quat = Quaternion(left_rotation_quat[-1], left_rotation_quat[0], left_rotation_quat[1], left_rotation_quat[2])
LEFT_ARM_EXTRINSICS = left_rotation_quat.transformation_matrix
LEFT_ARM_EXTRINSICS[0][-1] = left_translation[0]
LEFT_ARM_EXTRINSICS[1][-1] = left_translation[1]
LEFT_ARM_EXTRINSICS[2][-1] = left_translation[2]

# 0527 right arm calibration: Errors 0.0223576 m, 0.0120889 rad
# right_translation = [-1.0403, -0.44696, 0.39837] # x y z
# right_rotation_quat = [-0.75321, 0.1632, -0.15168, 0.6189] # (x, y, z, w)

# 0611 right arm calibration: Errors 0.0190123 m, 0.00931677 rad
right_translation = [-1.0229, -0.44106, 0.38526] # x y z
right_rotation_quat = [-0.75635, 0.16024, -0.15792, 0.61426] # (x, y, z, w)

# (w, x, y, z)
right_rotation_quat = Quaternion(right_rotation_quat[-1], right_rotation_quat[0], right_rotation_quat[1], right_rotation_quat[2])
RIGHT_ARM_EXTRINSICS = right_rotation_quat.transformation_matrix
RIGHT_ARM_EXTRINSICS[0][-1] = right_translation[0]
RIGHT_ARM_EXTRINSICS[1][-1] = right_translation[1]
RIGHT_ARM_EXTRINSICS[2][-1] = right_translation[2]
