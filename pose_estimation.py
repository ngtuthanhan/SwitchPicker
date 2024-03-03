import json
import numpy as np
from tqdm import tqdm
import math
import cv2

with open('./data/camera.json', 'r') as f:
    camera_info = json.load(f)

object_measurement = (11e-3, 19.8e-3, 6.4e-3)
def create_points(origin, theta_deg, distance):
    theta_rad = np.deg2rad(theta_deg)
    
    x2 = origin[0] + distance * np.cos(theta_rad)
    y2 = origin[1] - distance * np.sin(theta_rad)
    
    # slope = (y2 - origin[1]) / (x2 - origin[0])

    # perpendicular_slope = -1 / slope
    theta_rad += np.pi/2
    
    x3 = origin[0] + distance * math.cos(theta_rad)
    y3 = origin[1] - distance * math.sin(theta_rad) 
    
    return np.array([np.array(origin), [x2, y2], [x3, y3]])

def pose_estimation(camera_info, pick_points):

    intrinsics = np.array(camera_info["intrinsics"], dtype=np.float32)
    distortion = np.array(camera_info["distortion"], dtype=np.float32)

    translation_vector_list = []
    rotation_matrix_list = []
    axes = (
        np.array(
            [   
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                
            ]
        )
        * 0.005
    )

    for pick_point in pick_points:
        cx, cy, theta = pick_point
        image_pts = create_points((cx, cy), theta, 35)
        _, rvecs, tvecs = cv2.solveP3P(axes, image_pts, intrinsics, distortion, 5)
        rmat, _ = cv2.Rodrigues(rvecs[0])
        translation_vector_list.append(tvecs[0].flatten())
        rotation_matrix_list.append(rmat.flatten())

    return translation_vector_list, rotation_matrix_list

def save_pose_estimation_results(file_path, translation_vectors, rotation_matrices):
    with open(file_path, 'w') as file:
        for translation_vector, rotation_matrix in zip(translation_vectors, rotation_matrices):
            translation_str = ' '.join(map(str, translation_vector))
            rotation_str = ' '.join(map(str, rotation_matrix))
            file.write(f"{translation_str} {rotation_str} \n")

if __name__ == "__main__":
    with open('./data/test_ids.txt', 'r') as f:
        test_ids_str = f.read()
        test_ids_list = test_ids_str.split('\n')

    for test_id in tqdm(test_ids_list):
        with open(f'./result/part_1/{test_id}.txt', 'r') as f:
            pick_point_str = f.read()
            pick_point_list = pick_point_str.split('\n')
            pick_points = [[float(num) for num in pick_point.split()] for pick_point in pick_point_list]
        translation_vectors, rotation_matrices = pose_estimation(camera_info, pick_points)
        save_pose_estimation_results(f'./result/part_2/{test_id}.txt', translation_vectors, rotation_matrices)
