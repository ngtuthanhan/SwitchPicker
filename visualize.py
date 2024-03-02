"""Visualization helper functions."""
import typing as T
import cv2
import numpy as np
import pathlib
import json
import matplotlib.pyplot as plt

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


def imshow(image_bgr: np.ndarray):
    """Showing image."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()


def visualize_training_image(image_id: str):
    """Draw polygons on training image."""
    image_file = image_dir / f"{image_id}.png"
    image_bgr = cv2.imread(image_file.as_posix())

    polygons = annotation[image_id]["polygons"]
    for i, poly in enumerate(polygons):
        coord_x = poly["all_points_x"]
        coord_y = poly["all_points_y"]
        poly = np.array([coord_x, coord_y], dtype=np.int32).T.reshape(-1, 2)
        label = annotation[image_id]["labels"][i]
        if label == "top":
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red
        cv2.polylines(image_bgr, [poly], True, color, 2)

    imshow(image_bgr)


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

        cv2.line(image_bgr, origin_pt, image_pts[0], (255, 0, 0), 5)
        cv2.line(image_bgr, origin_pt, image_pts[1], (0, 255, 0), 5)
        cv2.line(image_bgr, origin_pt, image_pts[2], (0, 0, 255), 5)


def visualize_test_image(image_id: str):
    """Visualize the test image with pick frames."""
    image_file = image_dir / f"{image_id}.png"
    image_bgr = cv2.imread(image_file.as_posix())

    draw_pick_frames(
        image_bgr,
        annotation[image_id]["pick_points"],
        camera_info["intrinsics"],
        camera_info["distortion"],
    )

    imshow(image_bgr)


def visualize_all_training_images():
    """Visualize all training images."""
    for image_id in train_ids:
        visualize_training_image(image_id)


def visualize_all_test_images():
    """Visualize all test images."""
    for image_id in test_ids:
        visualize_test_image(image_id)


if __name__ == "__main__":
    # visualize_all_training_images()
    # visualize_all_test_images()
    visualize_training_image("100")
    visualize_test_image("102")
