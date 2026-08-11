import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

from apss import get_scalar_field
from visualization import plot_surface

def main():
    # loading data
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud("../point_clouds/bun_zipper.ply")
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)

    
    # processing datas
    print("Calculating scalar field...")
    samples = 128
    apss_field = get_scalar_field(pcd_points,pcd_normals, h=1.8,k=7,samples=samples)

    #rendering surface
    print("Rendering...")
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')

    plot_surface(pcd_points,ax,apss_field,color='crimson',samples=samples)

    plt.show()



if __name__ == '__main__':
    main()
    
