# Code modified from GeeksforGeeks and docs.scipy.org
import numpy as np
from scipy.optimize import minimize

# Define the 4-leaf rose
def rose_polar(theta):
    return np.sin(2 * theta)

def is_point_in_rose(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_rose = np.abs(rose_polar(theta))
    return r <= r_rose

def create_rectangle_vertices(x_center, y_center, width, height, angle_deg):
    angle_rad = np.radians(angle_deg)
    half_width = width / 2
    half_height = height / 2
    vertices = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_vertices = np.dot(vertices, rotation.T)
    translated_vertices = rotated_vertices + np.array([x_center, y_center])
    return translated_vertices

def calculate_intersection_area(parameters):
    import matplotlib.path as mpath
    x_center, y_center, angle_deg = parameters
    width = 1
    height = 1 / np.sqrt(2)
    rect_vertices = create_rectangle_vertices(x_center, y_center, width, height, angle_deg)
    n_points = 10000
    x_min, y_min = np.min(rect_vertices, axis=0)
    x_max, y_max = np.max(rect_vertices, axis=0)
    x_points = np.random.uniform(x_min, x_max, n_points)
    y_points = np.random.uniform(y_min, y_max, n_points)
    points = np.column_stack((x_points, y_points))
    rect_path = mpath.Path(rect_vertices)
    in_rectangle = rect_path.contains_points(points)
    intersection_count = 0
    rectangle_count = np.sum(in_rectangle)
    for i in range(n_points):
        if in_rectangle[i]:
            if is_point_in_rose(points[i, 0], points[i, 1]):
                intersection_count += 1
    rect_area = width * height
    if rectangle_count > 0:
        intersection_area = rect_area * (intersection_count / rectangle_count)
    else:
        intersection_area = 0
    return -intersection_area

def optimize_cutter_position():
    initial_guess = [0, 0, 0]
    bounds = [(-1.5, 1.5), (-1.5, 1.5), (0, 180)]
    result = minimize(
        calculate_intersection_area,
        initial_guess,
        method='Powell',
        bounds=bounds
    )
    return result


result = optimize_cutter_position()
if result.success:
    x_opt, y_opt, angle_opt = result.x
    print(f"Optimization succeeded")
    print(f"Optimal center: ({x_opt:.6f}, {y_opt:.6f})")
    print(f"Optimal angle: {angle_opt:.6f}°")
    print(f"Maximum intersection area: {-result.fun:.6f}")
else:
    print("Optimization failed:", result.message)

