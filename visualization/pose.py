"""Visualization helper functions."""
import typing as T
import cv2
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt

data_dir = (pathlib.Path(__file__).parent.parent / "data").resolve()
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

def imshow(image_bgr: np.ndarray):
    """Showing image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()

def draw_pick_frames(
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

    for frame in frames:
        tvec = np.array(frame["translation"])
        rmat = np.array(frame["rotation"])
        assert tvec.shape == (3,)
        assert rmat.shape == (3, 3)

        rvec = cv2.Rodrigues(rmat)[0]

        image_pts, _ = cv2.projectPoints(axes, rvec, tvec, cam_mtx, cam_dist)
        image_pts = image_pts.squeeze(1).astype(np.int32)
        origin_pt = image_pts[3]
        print(image_pts)

        cv2.line(image_bgr, origin_pt, image_pts[0], (255, 0, 0), 5)
        cv2.line(image_bgr, origin_pt, image_pts[1], (0, 255, 0), 5)
        cv2.line(image_bgr, origin_pt, image_pts[2], (0, 0, 255), 5)


def visualize_result(image_id: str):
    """Visualize the test image with pick frames."""
    image_file = image_dir / f"{image_id}.png"
    image_bgr = cv2.imread(image_file.as_posix())
    pose_result = (pathlib.Path(__file__).parent.parent / "result").resolve() / f"part_2/{image_id}.txt"
    pick_point_str = pose_result.read_text()
    pick_points = convert_to_list(pick_point_str)
    draw_pick_frames(
        image_bgr,
        pick_points,
        camera_info["intrinsics"],
        camera_info["distortion"],
    )

    imshow(image_bgr)


if __name__ == "__main__":
    visualize_result("102")
