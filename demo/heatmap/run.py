import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as mpl_polygon

def normalized_polygon(polygon):
  centroid = polygon.centroid
  # Get the coordinates of the centroid
  centroid_x, centroid_y = centroid.x, centroid.y

  # Create a new translated polygon by subtracting the centroid
  centered_polygon = Polygon([(x - centroid_x, y - centroid_y) for x, y in polygon.exterior.coords])
  return centered_polygon


from PIL import Image
import numpy as np
import os
import tqdm

# Replace 'mask_image_path.png' with the path to your black and white image
jacquard_dir = 'jacquard_mask/jacquard_mask'
files = os.listdir(jacquard_dir)
jacquard_polygons = []
for image_path in tqdm.tqdm(files[:2000]):

  # Open the image using PIL
  image = Image.open(os.path.join(jacquard_dir, image_path))

  # Convert the image to a NumPy array
  mask_array = np.array(image)
  indices = np.argwhere(mask_array.T > 0)
  polygon = Polygon(indices.tolist())
  jacquard_polygons.append(polygon)


ga_dir = 'ga_mask/ga_mask'
files = os.listdir(ga_dir)
ga_polygons = []
for mask_path in tqdm.tqdm(files[:2000]):

  # Open the image using PIL
  with open(os.path.join(ga_dir, mask_path), 'rb') as f:

  # Convert the image to a NumPy array
    mask_array = np.load(f)
    indices = np.argwhere(mask_array > 0)
    polygon = Polygon(indices.tolist())
    ga_polygons.append(polygon)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x_ga_ = []
y_ga_ = []
for p in ga_polygons:
  for point in p.exterior.coords:
    x_ga_.append(point[0]/2)
    y_ga_.append(point[1]/2)

jac_x_ = []
jac_y_ = []
for p in jacquard_polygons:
  for point in p.exterior.coords:
    jac_x_.append(point[0])
    jac_y_.append(point[1])

# Generate sample data for two maps (replace this with your actual data)
map1_data = np.random.normal(0, 1, size=(1000, 2))
map2_data = np.random.normal(2, 1, size=(1000, 2))

# Create two subplots side-by-side
plt.figure(figsize=(7, 3.5))

# Subplot 1
plt.subplot(1, 2, 1, aspect='equal')
plt.hist2d(x_ga_, y_ga_, bins=150, cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.title('Grasp-Anything')

# Subplot 2
plt.subplot(1, 2, 2, aspect='equal')
plt.hist2d(jac_x_, jac_y_, bins=150, cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.title('Jacquard')

# Specify the position for the colorbar
cax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]

# Show the colorbars for the 2D histograms
cbar = plt.colorbar(cax=cax)
cbar.set_label('Counts')

# Adjust layout to prevent overlap
# plt.tight_layout(w_pad=2.5)
# # Show the plots
# plt.colorbar()
# plt.show()
plt.savefig('shape_visualization.pdf', dpi=300, bbox_inches='tight')