import cv2
import numpy as np
import matplotlib.pyplot as plt


def unwrap_image(input_image):
    # Retrieve image height and diameter.
    image_height, diameter, _ = input_image.shape

    # Calculate radius
    radius = diameter / 2

    # Calculate arc length
    arc_length = int(np.pi * radius)

    # Calculate 2D maps using meshgrid.
    theta, y = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, arc_length, dtype='float32'),
                           np.arange(start=0, stop=image_height, step=1, dtype='float32'),
                           indexing='xy')
    # Calculate 2d map for x values using theta.
    x = (radius - 1) * np.sin(theta) + radius

    # return remap image, passing in both 2D matrices and using linear interpolation.
    return cv2.remap(input_image, x, y, cv2.INTER_LINEAR)


def remove_distortion(input_image, b_point):
    # Retrieve image height and diameter.
    image_height, diameter, _ = input_image.shape

    # Calculate center of image
    half_height, radius = (image_height / 2, diameter / 2)

    # Calculate a list of evenly spaced values between -b_point and b_point equal to image height
    b_values = np.linspace(-b_point, b_point, image_height)

    # Vectorize b values to pass into equation.
    b = np.array(b_values)[:, None]

    # Calculate a list of evenly spaced values between -half_height and half_height equal to image height
    y_values = np.linspace(-half_height, half_height, image_height)

    # Vectorize Y values to pass into equation.
    y_values = np.array(y_values)[:, None]

    # Calculate a list of evenly spaced values between -radius and radius equal to image width
    x_values = np.linspace(-radius, radius, diameter)

    # Generate 2D matrix containing repeated rows from 0 to image width (Diameter) using the mgrid.
    x_map = np.mgrid[0:image_height, 0:diameter][1]

    # Generate 2D matrix using the equation.
    y_map = (b / radius) * np.sqrt(radius ** 2 - x_values ** 2) + y_values + half_height - b

    # return remap image, passing in both 2D matrices and using linear interpolation.
    return cv2.remap(input_image, x_map.astype('float32'), y_map.astype('float32'), cv2.INTER_LINEAR)


image_path = 'image.bmp'  # Replace with your image path
image = cv2.imread(image_path)
e1 = cv2.getTickCount()
dst = remove_distortion(image, 46, )
unwrap_0 = unwrap_image(dst)
e2 = cv2.getTickCount()

time = (e2 - e1) / cv2.getTickFrequency()

print('Processing Time: ' + str(round(time * 1000, 2)) + 'ms')

cv2.imwrite("Output.png", unwrap_0)
plt.imshow(unwrap_0)
plt.axis('off')
plt.show()
