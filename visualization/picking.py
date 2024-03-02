import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_result(id):
    image = cv2.imread(f'./data/images/{id}.png')  
    with open(f'./result/part_1/{id}.txt', 'r') as f:
        pick_point_str = f.read()
        pick_point_list = pick_point_str.split('\n')
        pick_points = [pick_point.split(' ') for pick_point in pick_point_list]

    for point in pick_points:
        x, y, _ = point
        x, y = float(x), float(y)
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw a circle at each pick point

    # Visualize directions
    for point in pick_points:
        x, y, theta = point
        x, y, theta = float(x), float(y), float(theta)
        theta_rad = np.deg2rad(theta)
        
        x_end = int(x + 100 * np.cos(theta_rad)) 
        y_end = int(y - 100 * np.sin(theta_rad))  
        
        cv2.line(image, (int(x), int(y)), (x_end, y_end), (255, 0, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.show()

if __name__ == "__main__":
    visualize_result("102")