import numpy as np
from scipy.spatial import cKDTree

def get_scalar_field(points, normals, h, k, samples=25, batch_size=2000):
    # arguments:
    # points: the point cloud (loaded from a .ply file)
    # normals: the point cloud's normals (also loaded from a .ply file)
    # h: parameter that determines the radius of influence of a point upon its neighbours
    # k: for k nearest neighbours
    # samples: the resolution of the grid we are going to evaluate the scalar field
    # batch_size: number of points per batch

    x_range = np.linspace(np.min(points[:,0]), np.max(points[:,0]), samples)
    y_range = np.linspace(np.min(points[:,1]), np.max(points[:,1]), samples)
    z_range = np.linspace(np.min(points[:,2]), np.max(points[:,2]), samples)
    x, y, z = np.meshgrid(x_range, y_range, z_range)
    n = points.shape[0]

    sample_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    M = sample_points.shape[0]

    # creating a kdTree
    kd_tree = cKDTree(points)


    w_dom_flat = np.full(M, 100.0)

    # creating a vector that stores the radius of influence of all points
    k_dist, _ = kd_tree.query(sample_points,k=k)
    roi_all = np.maximum(k_dist[:, -1]*h, 1e-5)
    

    #processing each point separately and constructing the matrixes only for the points inside of the roi
    for i in range(M):
        p_i = sample_points[i]
        roi_i = roi_all[i]

        #getting all the points inside the roi
        idx = kd_tree.query_ball_point(p_i, r=roi_i)
        n_local = len(idx)

        if n_local < 4:
            continue

        local_points = points[idx]
        local_normals = normals[idx]

        
        dif = p_i - local_points
        dist_sq = np.sum(dif**2, axis=1)
        ratio_sq = dist_sq / (roi_i**2)
        weights = np.where(ratio_sq < 1.0, (1.0 - ratio_sq)**4, 0.0)

        #matrix construction
        
        b_local = np.zeros(4 * n_local)
        b_local[1::4] = local_normals[:, 0]
        b_local[2::4] = local_normals[:, 1]
        b_local[3::4] = local_normals[:, 2]
        
        #design matrix
        D_local = np.zeros((4 * n_local, 5))
        D_local[0::4, 0] = 1
        D_local[0::4, 1] = local_points[:, 0]
        D_local[0::4, 2] = local_points[:, 1]
        D_local[0::4, 3] = local_points[:, 2]
        D_local[0::4, 4] = np.sum(local_points**2, axis=1)

        D_local[1::4, 1] = 1
        D_local[1::4, 4] = 2 * local_points[:, 0]
        D_local[2::4, 2] = 1
        D_local[2::4, 4] = 2 * local_points[:, 1]
        D_local[3::4, 3] = 1
        D_local[3::4, 4] = 2 * local_points[:, 2]

        beta = 1e6 * (roi_i**2)
        
        #weight matrix
        W_diag = np.zeros(4 * n_local)
        W_diag[0::4] = weights
        W_diag[1::4] = beta * weights
        W_diag[2::4] = beta * weights
        W_diag[3::4] = beta * weights
        
        D_weighted = D_local * W_diag[:, np.newaxis] 
        A = D_local.T @ D_weighted
        A += 1e-6 * np.eye(5)

        b_hat = D_weighted.T @ b_local
        
        try:
            u = np.linalg.solve(A, b_hat)
        except np.linalg.LinAlgError:
            continue
            
        
        P_i = np.array([1, p_i[0], p_i[1], p_i[2], np.sum(p_i**2)])
        w_dom_flat[i] = np.dot(P_i, u)
    


    return w_dom_flat.reshape(x.shape)