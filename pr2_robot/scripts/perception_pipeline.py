#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import rospkg
import os.path
import xml.etree.ElementTree as ET
import re


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def passthrough_filter(cloud, filter_axis, axis_min, axis_max):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:
    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(10)
    threshold_scale_factor = .0001
    outlier_filter.set_std_dev_mul_thresh(threshold_scale_factor)
    cloud = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()

    # TODO: PassThrough Filter
    cloud = passthrough_filter(cloud, 'z', 0.6, 0.8)
    cloud = passthrough_filter(cloud, 'y', -0.5, 0.5)


    # TODO: RANSAC Plane Segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_objects = cloud.extract(inliers, negative=True)
    cloud_table =  cloud.extract(inliers, negative=False)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(2000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    cluster_color_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_color_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    pcl_msg = pcl_to_ros(cluster_color_cloud)

    # TODO: Publish ROS messages
    world_cloud_pub.publish(pcl_msg)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        cloud_object = cloud_objects.extract(pts_list)
        ros_cloud_object = pcl_to_ros(cloud_object)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cloud_object, using_hsv=True)
        normals = get_normals(ros_cloud_object)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_object[1])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = str(label)
        do.cloud = ros_cloud_object
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

def get_scene_num():
    path = os.path.join(rospkg.RosPack().get_path('pr2_robot'), 'launch', 'pick_place_project.launch')
    tree = ET.parse(path)
    root = tree.getroot()
    # find <include file="$(find gazebo_ros)/launch/empty_world.launch">
    scene_file = None
    for include in root.findall('include'):
        file_attr = include.get('file')
        if file_attr is not None and file_attr.endswith('launch/empty_world.launch'):
            for arg in include.findall('arg'):
                name_attr = arg.get('name')
                if name_attr is not None and name_attr == 'world_name':
                    value_attr = arg.get('value')
                    if value_attr is not None:
                        scene_file = value_attr
                    break
            break
    if scene_file is not None:
        m = re.match(r'.*/worlds/test(\d)\.world', scene_file)
        if m is not None:
            return int(m.group(1))
    return 0



# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    test_scene_num = Int32()
    test_scene_num.data = get_scene_num()

    # TODO: Initialize variables
    picking_list = []
    yaml_dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    # TODO: Parse parameters into individual variables
    for item in object_list_param:
        item_name = item['name']
        item_group = item['group']
        for obj in object_list:
            if item_name == obj.label:
                item_detected_object = obj
                break
        else:
            rospy.logwarn('Could not find item {}'.format(item_name))
            continue
        picking_list.append((item_detected_object, item_group))

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for obj, group in picking_list:
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]

        pick_pose = Pose()
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])

        # TODO: Create 'place_pose' for the object
        place_pose = Pose()

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()

        if group == 'red':
            place_pose.position.x = 0
            place_pose.position.y = 0.71
            place_pose.position.z = 0.605
            arm_name.data = 'left'
        else:
            place_pose.position.x = 0
            place_pose.position.y = -0.71
            place_pose.position.z = 0.605
            arm_name.data = 'right'

        object_name = String()
        object_name.data = obj.label

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        # # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    yaml_file_path = os.path.join(rospkg.RosPack().get_path('pr2_robot'), 'config', 'output_{:d}.yaml'.format(test_scene_num.data))
    send_to_yaml(yaml_file_path, yaml_dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('perception_pipeline', anonymous=True)

    # TODO: Create Subscribers
    world_cloud_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    world_cloud_pub = rospy.Publisher("/pr2/world/test", pc2.PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    rospy.loginfo('Loaded trained model.There are {} objects: {}'.format(len(model['classes']), ', '.join(model['classes'])))


    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()