import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm 


worker_env = {}

#initializing a dictionary with the relevant parameters to avoid passing these as arguments and use them by reference
def init_worker(points, normals, sample_points, roi_all):
    worker_env['points'] = points
    worker_env['normals'] = normals
    worker_env['sample_points'] = sample_points
    worker_env['roi_all'] = roi_all
    worker_env['kd_tree'] = cKDTree(points)


# function that will apply the core of the method to a chunk of the points
# meant to run in parallel 
def process_chunk(start, end):
    points = worker_env['points']
    normals = worker_env['normals']
    sample_points = worker_env['sample_points']
    roi_all = worker_env['roi_all']
    kd_tree = worker_env['kd_tree']

    w_chunk = np.full(end - start, 100.0)

    for i in range(start, end):
        p_i = sample_points[i]
        roi_i = roi_all[i]

        # finding the distance and index of the 100 nearest neighbours
        # 100 being an arbitrary cap on the number of neighbours to not impact performance
        dists, idx = kd_tree.query(p_i, k=100)

        valid_mask = dists < roi_i
        # filters the entries in idx that correspond to points satisfying the condition in valid_mask
        idx = idx[valid_mask]

        n_local = len(idx)
        if n_local < 4:
            continue



        local_points = points[idx]
        local_normals = normals[idx]

        # matrix that stores the distances between the current point and its locals
        dist_sq = dists[valid_mask] ** 2
        ratio_sq = dist_sq / (roi_i**2)
        # weight kernel
        weights = np.where(ratio_sq < 1.0, (1.0 - ratio_sq)**4, 0.0)

        # matrix construction based on the reference paper
        
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
        w_chunk[i - start] = np.dot(P_i, u)

    return start, end, w_chunk


def get_scalar_field(points, normals, h, k, samples=25):
    # arguments:
        # points: the point cloud (loaded from a .ply file)
        # normals: the point cloud's normals (also loaded from a .ply file)
        # h: parameter that determines the radius of influence of a point upon its neighbours
        # k: for k nearest neighbours
        # samples: the resolution of the grid we are going to evaluate the scalar field

    # making the grid that we will use to evaluate the scalar field
    x_range = np.linspace(np.min(points[:,0]), np.max(points[:,0]), samples)
    y_range = np.linspace(np.min(points[:,1]), np.max(points[:,1]), samples)
    z_range = np.linspace(np.min(points[:,2]), np.max(points[:,2]), samples)
    x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    sample_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    M = sample_points.shape[0]

    kd_tree = cKDTree(points)
    w_dom_flat = np.full(M, 100.0)

    print("Pre-processing the radius of influence (ROI)...")
    k_dist, _ = kd_tree.query(sample_points, k=k)
    roi_all = np.maximum(k_dist[:, -1] * h, 1e-5)

    num_cores = multiprocessing.cpu_count()
    
    # divide the total points in 100 chunks
    num_chunks = 100
    chunk_size = int(np.ceil(M / num_chunks))
    
    futures = []
    
    with ProcessPoolExecutor(max_workers=num_cores,
                            initializer=init_worker,
                            initargs=(points,normals,sample_points,roi_all)) as executor:
        # assigns point chunks to the cores
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(M, (i + 1) * chunk_size)
            if start < M:    
                futures.append(
                    executor.submit(process_chunk, start, end   )
                )

        # upon completion, each core's chunk will be pieced together in the w_chunk matrix
        with tqdm(total=M, desc="Computing the scalar fields...", unit="pts") as pbar:
            
            for future in as_completed(futures):
                start, end, w_chunk = future.result()
                w_dom_flat[start:end] = w_chunk
                
                pbar.update(end - start)

    return w_dom_flat.reshape(x.shape)