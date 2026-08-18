import open3d as o3d
import numpy as np




def calculate_apss_error(original_ply, apss_ply):
    original_pcd = o3d.io.read_point_cloud(original_ply)

    apss_pcd = o3d.io.read_point_cloud(apss_ply)

    dists = original_pcd.compute_point_cloud_distance(apss_pcd)
    dists = np.asarray(dists)

    mean_error = np.mean(dists)
    rmse = np.mean(np.sqrt(dists**2))
    haussdorf = np.max(dists)

    print(f"{apss_ply} error values:")
    print(f"mean error: {mean_error}")
    print(f"rmse: {rmse}")
    print(f"haussdorf: {haussdorf}")


