import os
import torch
import config
import numpy as np
from libs.body_model.joint_names import SMPLH_JOINT_NAMES
from tools.rotation_conversions import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle

def get_pose_data_from_file(pose_info, applied_rotation=None, output_rotation=False):
	"""
	Load pose data and normalize the orientation.

	Args:
		pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]
		applied_rotation: rotation to be applied to the pose data. If None, the
			normalization rotation is applied.
		output_rotation: whether to output the rotation performed for
			normalization, in addition of the normalized pose data.

	Returns:
		pose data, torch.tensor of size (1, n_joints*3), all joints considered.
		(optional) R, torch.tensor representing the rotation of normalization
	"""

	# load pose data
	assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
	
	if pose_info[0] == "AMASS":
		dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
		pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
		pose = torch.as_tensor(pose).to(dtype=torch.float32)

	# normalize the global orient
	initial_rotation = pose[:1,:].clone()
	if applied_rotation is None:
		euler = matrix_to_euler_angles(axis_angle_to_matrix(initial_rotation), 'ZYX')
		# thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
		# zeros = torch.zeros_like(thetaz)
		euler[:, -1] = 0
		pose[0:1,:] = matrix_to_axis_angle(euler_angles_to_matrix(euler, 'ZYX'))
	else:
		pose[0:1,:] = matrix_to_axis_angle(axis_angle_to_matrix(applied_rotation) @ axis_angle_to_matrix(initial_rotation))
		# pose[0:1,:] = roma.rotvec_composition((applied_rotation, initial_rotation))
	if output_rotation:
		# a = A.u, after normalization, becomes a' = A'.u
		# we look for the normalization rotation R such that: a' = R.a
		# since a = A.u ==> u = A^-1.a
		# a' = A'.u = A'.A^-1.a ==> R = A'.A^-1
		# R = roma.rotvec_composition((pose[0:1,:], roma.rotvec_inverse(initial_rotation)))
		R = matrix_to_axis_angle(axis_angle_to_matrix(pose[0:1,:]) @ torch.inverse(axis_angle_to_matrix(initial_rotation)))
		return pose.reshape(1, -1), R
	
	return pose.reshape(1, -1)

def pose_data_as_dict(pose_data):
	"""
	Args:
		pose_data, torch.tensor of shape (*, n_joints*3) or (*, n_joints, 3),
			all joints considered.

	Returns:
		dict
	"""
	# reshape to (*, n_joints*3) if necessary
	if len(pose_data.shape) == 3:
		# shape (batch_size, n_joints, 3)
		pose_data = pose_data.flatten(1,2)
		
	if pose_data.shape[1] == 66:
		return {"root_orient":pose_data[:,:3],
				"pose_body":pose_data[:,3:66],
				"pose_hand":None}

	# provide as a dict, with the expected keys
	return {"root_orient":pose_data[:,:3],
			"pose_body":pose_data[:,3:66],
			"pose_hand":pose_data[:,66:]}

def get_smplh_groups(group_num, with_root=True, unique=False, return_name=False):
    if group_num == 7:
        joints_index = list(range(52))
        if not unique:
            group_names = [
                ['pelvis','left_hip','left_knee','left_ankle','left_foot'],
                ['pelvis','right_hip','right_knee','right_ankle','right_foot'],
                ['pelvis','spine1','spine2','spine3','neck','head'],
                ['pelvis','spine1','spine2','spine3','left_collar','left_shoulder','left_elbow','left_wrist'],
                ['pelvis','spine1','spine2','spine3','right_collar','right_shoulder','right_elbow','right_wrist'],
                ['left_wrist','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3'],
                ['right_wrist','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'],
            ]
        else:
            group_names = [
                ['pelvis','left_hip','left_knee','left_ankle','left_foot'],
                ['pelvis','right_hip','right_knee','right_ankle','right_foot'],
                ['pelvis','spine1','spine2','spine3','neck','head'],
                ['left_collar','left_shoulder','left_elbow','left_wrist'],
                ['right_collar','right_shoulder','right_elbow','right_wrist'],
                ['left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3'],
                ['right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'],
            ]
    elif group_num == 5:
        joints_index = list(range(22))
        if not unique:
            group_names = [
                ['pelvis','left_hip','left_knee','left_ankle','left_foot'],
                ['pelvis','right_hip','right_knee','right_ankle','right_foot'],
                ['pelvis','spine1','spine2','spine3','neck','head'],
                ['pelvis','spine1','spine2','spine3','left_collar','left_shoulder','left_elbow','left_wrist'],
                ['pelvis','spine1','spine2','spine3','right_collar','right_shoulder','right_elbow','right_wrist'],
            ]
        else:
            group_names = [
                ['left_hip','left_knee','left_ankle','left_foot'],
                ['right_hip','right_knee','right_ankle','right_foot'],
                ['pelvis','spine1','spine2','spine3','neck','head'],
                ['left_collar','left_shoulder','left_elbow','left_wrist'],
                ['right_collar','right_shoulder','right_elbow','right_wrist'],
            ]
    else:
        raise ValueError

    if not with_root:
        for i in range(len(group_names)):
            if "pelvis" in group_names[i]:
                group_names[i].remove("pelvis")
        joints_index.remove(0)

    group_index = [[SMPLH_JOINT_NAMES.index(name) for name in group_name] for group_name in group_names]

    if return_name:
        return joints_index, group_index, group_names

    return joints_index, group_index