#!/usr/bin/env python
import os, sys
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
import rospy, rospkg, tf
import pypcd
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import tf.transformations as tft

# convert ros data to kitti odometry format
class ros2odom:

  def __init__(self):
    # get incoming msgs
    self.lidar_sub = rospy.Subscriber("~lidar_in", PointCloud2, self.lidar_callback)
    self.odom_sub  = rospy.Subscriber("~odom_in",  Odometry,    self.odom_callback)
    # what sequence is being created?
    #  0-21 exist in kitti dataset. Best to go upwards of 70
    self.sequence = rospy.get_param('~sequence')

    # every scan is relative to first scan (origin)
    self.origin = None

    self.counter = 0

    # different means for darknet config
    self.x_mean = []
    self.y_mean = []
    self.z_mean = []
    self.intensity_mean = []
    self.range_mean = []

    # create file to save data
    rospack   = rospkg.RosPack()
    self.path = rospack.get_path('point_labeler')
    self.path = Path(self.path, 'dataset/sequences/{:02d}'.format(self.sequence))
    # does path exist?
    if not os.path.isdir(str(self.path)):
      os.makedirs(str(self.path))
    if not os.path.isdir(str(self.path) + '/velodyne'):
      os.mkdir(str(self.path) + '/velodyne')
    if not os.path.isdir(str(self.path) + '/labels'):
      os.mkdir(str(self.path) + '/labels')
    rospy.loginfo('\n \033[92m creating dataset in: {} \033[0m'.format(self.path))

    self.pose_file  = Path(self.path, 'poses').with_suffix('.txt')
    self.calib_file = Path(self.path, 'calib').with_suffix('.txt')
    self.stats_file = Path(self.path, 'additional_stats_{}'.format(self.sequence)).with_suffix('.txt')
    self.pose_filename = open(str(self.pose_file), 'w')

    # write dummy calibration file.
    #  Note: the actual calibration should be irrelevant for our purposes, since we have poses in the Velodyne coordinate system.
    #  if you want to have the "real" thing, one has to transform the poses into the camera coordinate system via Tr.
    calib_file = open(str(self.calib_file), 'w')
    calib_file.write("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib_file.write("P1: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib_file.write("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib_file.write("P3: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib_file.write("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib_file.close()


  # keep collecting (and overwriting) odom msg for latest
  def odom_callback(self, msg):
    self.odom_msg = msg.pose.pose


  # for every pointcloud, get the pose
  def lidar_callback(self, msg):
    # check if odom msg is populated
    if not self.odom_msg:
      pass

    rospy.logwarn('size: {}'.format(msg.height*msg.width))

    # get pose and populate file
    # translation
    t_matrix = tft.translation_matrix([self.odom_msg.position.x, self.odom_msg.position.y, self.odom_msg.position.z])
    # rotation
    r_matrix = tft.quaternion_matrix([self.odom_msg.orientation.x, self.odom_msg.orientation.y, self.odom_msg.orientation.z, self.odom_msg.orientation.w])
    # create a matix
    mat = np.dot(t_matrix, r_matrix)
    # set origin (first scan)
    if self.origin is None: self.origin = np.linalg.inv(mat)
    # transform wrt origin
    mat = np.dot(self.origin, mat)
    rospy.logdebug('mat:\n{}'.format(mat))
    # write to file
    self.pose_filename.write(' '.join([str(v) for v in mat.reshape(-1)[:12]]) + '\n')

    # convert scan to .bin format
    pc = pypcd.PointCloud.from_msg(msg)
    x = pc.pc_data['x']
    y = pc.pc_data['y']
    z = pc.pc_data['z']  # ~0.75m diference between bolt and kitti lidar position
    range_i = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    intensity = pc.pc_data['intensity']

    arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
    arr[::4] = x
    arr[1::4] = y
    arr[2::4] = z
    arr[3::4] = intensity

    filename = Path(self.path, 'velodyne', '{:06d}.bin'.format(msg.header.seq))
    arr.astype('float32').tofile(str(filename))
    rospy.loginfo('saving pointcloud as \'{}.bin\''.format(filename.stem))

    # update means
    self.x_mean.append(np.mean(x))
    self.y_mean.append(np.mean(y))
    self.z_mean.append(np.mean(z))
    self.intensity_mean.append(np.mean(intensity))
    self.range_mean.append(np.mean(range_i))

    rospy.loginfo('sequence #{}'.format(self.counter))
    rospy.logwarn('range_mean:{:.2f} x_mean:{:.2f} y_mean:{:.2f} z_mean:{:.2f} intensity_mean:{:.2f}'.format(np.mean(self.range_mean), np.mean(self.x_mean), np.mean(self.y_mean), np.mean(self.z_mean), np.mean(self.intensity_mean)))
    rospy.logwarn('range_std:{:.2f} x_std:{:.2f} y_std:{:.2f} z_std:{:.2f} intensity_std:{:.2f}'.format(np.std(self.range_mean), np.std(self.x_mean), np.std(self.y_mean), np.std(self.z_mean), np.std(self.intensity_mean)))

    # save to file
    mean_list = ['mean', np.mean(self.range_mean), np.mean(self.x_mean), np.mean(self.y_mean), np.mean(self.z_mean), np.mean(self.intensity_mean)]
    std_list  = ['std', np.std(self.range_mean), np.std(self.x_mean), np.std(self.y_mean), np.std(self.z_mean), np.std(self.intensity_mean)]
    stats = np.stack((mean_list, std_list), axis=0)
    np.savetxt(self.stats_file, stats, fmt='%s')

    self.counter += 1

# standard ROS broilerplate
if __name__ == '__main__':
  rospy.init_node('ros_2_odom')
  ros2odom()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
