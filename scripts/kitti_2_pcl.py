#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

pcd_header = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z label
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii\n"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser('./kitti_2_pcl.py')
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to convert. No Default',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=int,
      required=True,
      help='Sequence to convert. No Default',
  )
  FLAGS, unparsed = parser.parse_known_args()
  # sequence name must be in format 02d
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  print('Analyzing')
  print('  Dataset', FLAGS.dataset)
  print('  Sequence', FLAGS.sequence)

  # check if correct directory structure exists
  scan_path = Path(FLAGS.dataset, "sequences", FLAGS.sequence, "velodyne")
  if not scan_path.is_dir():
    print("Scans folder doesn't exist! Exiting...")
    sys.exit()

  label_path = Path(FLAGS.dataset, "sequences", FLAGS.sequence, "labels")
  if not label_path.is_dir():
    print("Labels folder doesn't exist! Exiting...")
    sys.exit()

  # if pcds dir doesn't exist, make one
  pcd_path = Path(FLAGS.dataset, 'sequences', FLAGS.sequence, 'pcds')
  if not pcd_path.exists():
    pcd_path.mkdir()

  # loop thropuh each file
  for file in scan_path.glob('*.bin'):
    # get scan and corresponding labels file
    scan_name  = Path(scan_path, file.name)
    label_name = Path(label_path, scan_name.with_suffix('.label').name)

    # read pointcloud
    scan = np.fromfile(scan_name, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # print('loaded scan with shape {}'.format(scan.shape))
    # scan attributes
    points     = scan[:, 0:3] # get xyz
    remissions = scan[:, 3]   # get remission
    # print(points)

    # read label file
    label = np.fromfile(label_name, dtype=np.uint32)
    label = label.reshape((-1))
    # print('loaded label with shape {}'.format(label.shape))
    # print(label)

    # create pcd dataframe
    df = pd.DataFrame(points, columns = ['x','y','z'])
    df['label'] = label
    # print(df.head())

    # write pcd
    pcd_file = Path(pcd_path, scan_name.stem).with_suffix('.pcd')
    print('writing file {}'.format(str(pcd_file)))
    pcd = open(str(pcd_file), "w")
    # write header with num points
    pcd.write(pcd_header.format(len(points), len(points)))
    # save x, y, z and label
    df.to_csv(pcd, sep=' ', header=False, index=False)
    pcd.close()

    # sys.exit()
