import json
import numpy as np
from tqdm import tqdm

with open('./data/camera.json', 'r') as f:
    camera_info = json.load(f)

object_measurement = (11e-3, 19.8e-3, 6.4e-3)

def pose_estimation(object_measurement, camera_info, pick_points):
    height, width, depth = object_measurement

    intrinsics = np.array(camera_info["intrinsics"])
    distortion = np.array(camera_info["distortion"])

    translation_vector_world_list = []
    rotation_matrix_list = []

    for pick_point in pick_points:
        cx, cy, theta = pick_point

        rotation_matrix = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
                                    [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
                                    [0, 0, 1]])

        image_point = np.array([[cx], [cy], [1]])
        camera_point = np.dot(np.linalg.inv(intrinsics), image_point)
        camera_point = np.vstack((camera_point, [[1]]))

        translation_vector = np.array([[depth * np.cos(np.radians(theta))],
                                       [depth * np.sin(np.radians(theta))],
                                       [height / 2]])

        translation_vector_world = np.dot(rotation_matrix, translation_vector)

        translation_vector_world_list.append(translation_vector_world.flatten())
        rotation_matrix_list.append( rotation_matrix.flatten())

    return translation_vector_world_list, rotation_matrix_list

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
            pick_points = [pick_point.split(' ') for pick_point in pick_point_list]
            pick_points = [(float(pick_point[0]), float(pick_point[1]), float(pick_point[2])) for pick_point in pick_point_list]

        translation_vectors, rotation_matrices = pose_estimation(object_measurement, camera_info, pick_points)
        save_pose_estimation_results(f'./result/part_2/{test_id}.txt', translation_vectors, rotation_matrices)
