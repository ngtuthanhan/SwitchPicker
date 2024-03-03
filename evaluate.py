import numpy as np
from scipy.optimize import linear_sum_assignment
import typing as T
import cv2
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = (pathlib.Path(__file__).parent / "data").resolve()
image_dir = data_dir / "images"
annotation_file = data_dir / "annotation.json"
camera_info_file = data_dir / "camera.json"
train_ids_file = data_dir / "train_ids.txt"
test_ids_file = data_dir / "test_ids.txt"

annotation = json.loads(annotation_file.read_text())
camera_info = json.loads(camera_info_file.read_text())
train_ids = train_ids_file.read_text().splitlines()
test_ids = test_ids_file.read_text().splitlines()

def convert_to_list(input_str):
    lines = input_str.strip().split("\n")
    output_list = []

    for line in lines:
        elements = line.split()
        translation = [float(elements[0]), float(elements[1]), float(elements[2])]
        rotation = [
            [float(elements[3]), float(elements[4]), float(elements[5])],
            [float(elements[6]), float(elements[7]), float(elements[8])],
            [float(elements[9]), float(elements[10]), float(elements[11])]
        ]
        output_list.append({
            "translation": translation,
            "rotation": rotation
        })

    return output_list


def get_pick_frames(
    image_bgr: np.ndarray,
    frames: T.Sequence[T.Dict[str, list]],
    cam_mtx,
    cam_dist,
    length: float = 0.005,
):
    """Draw pick frames."""
    cam_mtx = np.array(cam_mtx, dtype=np.float32)
    cam_dist = np.array(cam_dist, dtype=np.float32)
    
    axes = (
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        * length
    )
    points = []
    for frame in frames:
        tvec = np.array(frame["translation"])
        rmat = np.array(frame["rotation"])
        assert tvec.shape == (3,)
        assert rmat.shape == (3, 3)
        rvec = cv2.Rodrigues(rmat)[0]
      
        image_pts, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, cam_dist)
        image_pts = image_pts.squeeze(1).astype(int)

        origin_pt = image_pts[3]

        perpendicular_direction = np.array([origin_pt[0]-image_pts[0][0], origin_pt[1]-image_pts[0][1]], dtype=float)  # Specify dtype=float
        perpendicular_direction /= np.linalg.norm(perpendicular_direction)
        theta = np.arctan2(perpendicular_direction[0], perpendicular_direction[1]) * 180 / np.pi
        points.append((origin_pt[0],  origin_pt[1], theta))
    return points


def get_gt_test_image(image_id: str):
    """Visualize the test image with pick frames."""
    image_file = image_dir / f"{image_id}.png"
    image_bgr = cv2.imread(image_file.as_posix())

    return get_pick_frames(
        image_bgr,
        annotation[image_id]["pick_points"],
        camera_info["intrinsics"],
        camera_info["distortion"],
    )

def get_pred_test_image(image_id: str):
    """Visualize the test image with pick frames."""
    image_file = image_dir / f"{image_id}.png"
    image_bgr = cv2.imread(image_file.as_posix())
    pose_result = (pathlib.Path(__file__).parent / "result").resolve() / f"part_2/{image_id}.txt"
    pick_point_str = pose_result.read_text()
    pick_points = convert_to_list(pick_point_str)

    return get_pick_frames(
        image_bgr,
        pick_points,
        camera_info["intrinsics"],
        camera_info["distortion"],
    )



def calculate_mae(ground_truth_points, predicted_points):
    n_gt = len(ground_truth_points)
    n_pred = len(predicted_points)

    errors = np.zeros((n_gt, n_pred))
    errors_point_only = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            errors_point_only[i, j] = abs(ground_truth_points[i][0] - predicted_points[j][0]) + \
                           abs(ground_truth_points[i][1] - predicted_points[j][1])

            errors[i, j] = abs(ground_truth_points[i][0] - predicted_points[j][0]) + \
                           abs(ground_truth_points[i][1] - predicted_points[j][1]) + \
                           abs(ground_truth_points[i][2] - predicted_points[j][2])

    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(errors_point_only)

    # Calculate total error
    total_error = 0
    for i, j in zip(row_ind, col_ind):
        total_error += errors[i, j]

    # Calculate MAE
    mae = total_error / n_gt

    return mae

def normalize_values(data):
    normalized_data = []
    for x, y, theta in data:
        normalized_x = x / 1920
        normalized_y = y / 1080
        normalized_theta = theta / 360
        normalized_data.append((normalized_x, normalized_y, normalized_theta))
    return normalized_data

def compute_all_test_images():
    avg_mae = []
    for image_id in tqdm(test_ids):
        gt_points = get_gt_test_image(image_id)
        gt_points = normalize_values(gt_points)
        pred_points = get_pred_test_image(image_id)
        pred_points = normalize_values(pred_points)
        mae = calculate_mae(gt_points, pred_points)
        avg_mae.append(mae)
    return np.mean(avg_mae)



if __name__ == "__main__":
    # visualize_all_training_images()
    # visualize_all_test_images()
    # visualize_training_image("100")
    print(compute_all_test_images())
