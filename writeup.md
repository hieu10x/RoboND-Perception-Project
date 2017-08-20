[//]: # (Image References)
[ex1_outliers]: ./misc/ex1_outliers.png
[ex1_inliers]: ./misc/ex1_inliers.png
[ex2_euclidean_cloud]: ./misc/ex2_euclidean_cloud.png
[ex3]: ./misc/ex3.png

## Project: Perception Pick & Place
---

### Writeup / README

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Exercise 1: Pipeline for filtering and RANSAC plane fitting implemented.
```Python
# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')

# Voxel Grid filter
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

# PassThrough filter
passthrough = cloud.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)

# RANSAC plane segmentation
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance 
# for segmenting the table
max_distance = 0.005
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)

# Extract outliers
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
filename = 'extracted_outliers.pcd'
pcl.save(extracted_outliers, filename)
```
![Extracted inliers cloud][ex1_inliers]
*Extracted inliers cloud*

![Extracted outliers cloud][ex1_outliers]
*Extracted outliers cloud*

#### 2. Exercise 2: Pipeline including clustering for segmentation implemented.
```Python
#!/usr/bin/env python

# Import modules
from pcl_helper import *
import rospy

# TODO: Define functions as required
def passthrough_filter(cloud, filter_axis, axis_min, axis_max):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()

    # PassThrough Filter
    cloud = passthrough_filter(cloud, 'z', 0.6, 1.1)
    cloud = passthrough_filter(cloud, 'y', -2.0, -1.4)

    # RANSAC Plane Segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_objects = cloud.extract(inliers, negative=True)
    cloud_table =  cloud.extract(inliers, negative=False)


    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)


    # Publish ROS messages
    cluster_pub.publish(ros_cluster_cloud)
    objects_pub.publish(ros_cloud_objects)
    table_pub.publish(ros_cloud_table)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
```
![Euclidean clouds][ex2_euclidean_cloud]
*Euclidean clouds*

#### 3. Exercise 3: Features extracted and SVM trained. Object recognition implemented.
##### sensor_stick/scripts/capture_features.py:
Changed the used color space to hsv
```Python
chists = compute_color_histograms(sample_cloud, using_hsv=True)
```
Changed the number of samples for each object from `5` to `50`
```Python
    for model_name in models:
        spawn_model(model_name)

        for i in range(50):
            # ...
```

##### sensor_stick/src/sensor_stick/features.py:
```Python
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *

print('Load this module')

def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # Compute histograms
    bins = 16
    channel_1_hist = np.histogram(channel_1_vals, bins=bins, range=(0, 256))
    channel_2_hist = np.histogram(channel_2_vals, bins=bins, range=(0, 256))
    channel_3_hist = np.histogram(channel_3_vals, bins=bins, range=(0, 256))

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0])).astype(np.float64)

    normed_features = hist_features / np.sum(hist_features)
    return normed_features


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute histograms of normal values (just like with color)
    bins = 16
    norm_x_hist = np.histogram(norm_x_vals, bins=bins, range=(-1.0, 1.0))
    norm_y_hist = np.histogram(norm_y_vals, bins=bins, range=(-1.0, 1.0))
    norm_z_hist = np.histogram(norm_z_vals, bins=bins, range=(-1.0, 1.0))

    # Concatenate and normalize the histograms
    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    normed_features = hist_features / np.sum(hist_features)

    return normed_features
```
##### sensor_stick/scripts/train_svm.py:
Kernel `linear` of the Support Vector Machine Classifier seems to work well. Hence no change has been made.

##### Results:
![Training results][ex3]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

Please refer to the code in the file [pr2_robot/scripts/perception_pipeline.py](pr2_robot/scripts/perception_pipeline.py).

#### Elaboration
- Feature capturing and training use the same code as *Exercise 3* except the ones mentioned below.
- `sensor_stick/scripts/capture_features.py` script is modified to capture features of 8 objects `biscuits, soap, soap2, book, glue, sticky_notes, snacks, eraser`.
- The number of samples for each object increases to `500`. This takes a significant amount of time to finish but it is important to raise the detection accuracy.
- At first, *brf* kernel of SVC was used and it gave a high accuracy score with cross validation. However, when applying to the test scenes, it did not give a very good detection of the trained objects. *linear* kernel was then chosen and by trials and errors, the following parameters were used:
```Python
clf = svm.SVC(kernel='linear', C=0.15)
```
- The parameters were *manually* selected and probably not the best set of parameters. An *automated* search, for example, using `sklearn.model_selection.GridSearchCV` could have been used instead to save the laborious work of trying different values of the parameters.
