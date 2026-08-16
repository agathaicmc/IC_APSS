import open3d as o3d
import numpy as np


from apss import get_scalar_field
from export import process_scalar_field

def main():
    # loading data
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud("point_clouds/dragon_vrip.ply")
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)

    file_name = input("Type the name of the file that will be created\n")
    # processing data
    samples = 128
    apss_field = get_scalar_field(pcd_points,pcd_normals, h=2.1,k=7,samples=samples)

    #rendering surface
    print("Exporting...")

    process_scalar_field(pcd_points, apss_field, samples=samples, file_name=file_name)



if __name__ == '__main__':
    main()
    
