from apss import get_scalar_field
import open3d as o3d
import numpy as np
import time


files = ["bun_zipper.ply", "Armadillo.ply", "dragon_vrip.ply", "happy_vrip.ply", "xyzrgb_dragon.ply", "lucy.ply"]
time_values = []


for file_name in files:
    pcd = o3d.io.read_point_cloud("point_clouds/" + file_name)
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)
    print("Processing " + file_name + "...")
    start_time = time.time()
    get_scalar_field(pcd_points, pcd_normals, h=1.8, k=7, samples=128)
    end_time = time.time()
    time_values.append(end_time - start_time)
    print(time_values)














