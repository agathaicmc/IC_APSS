import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_scalar_field(points, normals, h, k, samples=20):
    # arguments:
    # points: the point cloud (loaded from a .ply file)
    # normals: the point cloud's normals (also loaded from a .ply file)
    # h: parameter that determines the radius of influence of a point upon its neighbours
    # k: for k nearest neighbours
    base_range = np.linspace(np.min(points), np.max(points), samples)
    x, y, z = np.meshgrid(base_range, base_range, base_range)
    n = points.shape[0]

    sample_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    M = sample_points.shape[0]

    # making the matrixes described in the paper of reference
    b = np.zeros(4 * n)
    b[1::4] = normals[:, 0]
    b[2::4] = normals[:, 1]
    b[3::4] = normals[:, 2]

    D = np.zeros((4 * n, 5))
    D[0::4, 0] = 1                          
    D[0::4, 1] = points[:, 0]               
    D[0::4, 2] = points[:, 1]               
    D[0::4, 3] = points[:, 2]               
    D[0::4, 4] = np.sum(points**2, axis=1)  

    D[1::4, 1] = 1
    D[1::4, 4] = 2 * points[:, 0]
    D[2::4, 2] = 1
    D[2::4, 4] = 2 * points[:, 1]
    D[3::4, 3] = 1
    D[3::4, 4] = 2 * points[:, 2]

    # creating a kdTree
    kd_tree = cKDTree(points)

    # an array that contains the distance to all knn of the i'th sample point in the i'th row
    k_dist = kd_tree.query(sample_points,k=k)[0]

    # an array that contains the radius of influence of each point in a similar way to the k_dist array
    roi = np.maximum(k_dist[:,-1] * h, 1e-5)

    # the distances matrix stores the distance from each point in samplepoints to each single point in the point cloud
    dif = sample_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(dif, axis=2)

    ratio = distances / roi[:,np.newaxis]
    # || p_i - x ||/roi(x)
    # ratio: for each point, the distance from a point (x) in the point cloud to a sample point (p_i), divided by the roi of x
    
    # generic weighting function taken from the reference paper
    weights = np.where(ratio < 1, (1.0 - ratio**2)**4, 0.0)

    # gets only samples whose weight are larger than 1e-8
    active_samples = np.sum(weights > 1e-8, axis=1)

    # TO-DO: comment the shit below
    beta = 1e6 * roi**2
    
    W_diag = np.zeros((M, 4 * n))
    W_diag[:, 0::4] = weights
    W_diag[:, 1::4] = beta[:,np.newaxis] * weights
    W_diag[:, 2::4] = beta[:,np.newaxis] * weights
    W_diag[:, 3::4] = beta[:,np.newaxis] * weights
    
    D_weighted = D[np.newaxis, :, :] * W_diag[:, :, np.newaxis]
    A = np.matmul(D.T, D_weighted)
    A += 1e-6 * np.eye(5) 

    b_hat = (W_diag * b) @ D
    u_batch = np.linalg.solve(A, b_hat)

    P_poly = np.zeros((M, 5))
    P_poly[:, 0] = 1
    P_poly[:, 1] = sample_points[:, 0]
    P_poly[:, 2] = sample_points[:, 1]
    P_poly[:, 3] = sample_points[:, 2]
    P_poly[:, 4] = np.sum(sample_points**2, axis=1)

    dom_w_flat = np.sum(P_poly * u_batch, axis=1)

    dom_w_flat[active_samples < 4] = 100.0

    return dom_w_flat.reshape(x.shape())

def plot_surface(points, ax, scalar_field, color, samples=20):
    min_range = np.min(points)
    max_range = np.max(points)
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=0.0)
    
    spacing = (max_range - min_range) / (samples - 1)
    verts = min_range + verts * spacing
    
    mesh = Poly3DCollection(verts[faces], alpha=0.8, edgecolor='none')
    mesh.set_facecolor(color)
    mesh.set_edgecolor('black')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)
        
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)
    ax.set_zlim(min_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False




# reads the point clouds from a .ply file using the open3d library which was imported in line 1
pcd = o3d.io.read_point_cloud("point_clouds/bun_zipper_mesh.ply")
pcd_points = np.asarray(pcd.points)
pcd_normals = np.asarray(pcd.normals)

apss_field = get_scalar_field(pcd_points,pcd_normals,h=1.8,k=7)


fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 2, 1, projection='3d') 

plot_surface(pcd_points, ax, apss_field, color='crimson')

o3d.visualization.draw_geometries([pcd], point_show_normal=True)