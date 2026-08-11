import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

def plot_surface(points, ax, scalar_field, color, samples=25):

    #finding the boundaries of the point cloud
    min_x, max_x = np.min(points[:,0]), np.max(points[:,0])
    min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
    min_z, max_z = np.min(points[:,2]), np.max(points[:,2])
    # picking the biggest and smallest to keep the cubic image aspect
    global_min = min(min_x, min_y, min_z)
    global_max = max(max_x, max_y, max_z)
    try:

        verts, faces, _, _ = measure.marching_cubes(scalar_field, level=0.0)

        spacing = (global_max - global_min) / (samples - 1)
        verts = global_min + verts * spacing
        
        mesh = Poly3DCollection(verts[faces], alpha=0.8, edgecolor='none')
        mesh.set_facecolor(color)
        mesh.set_edgecolor('black')
        mesh.set_linewidth(0.2)
        ax.add_collection3d(mesh)

        export_mesh(verts, faces)

    except ValueError:
        print("Warning: no zero level surface found")    
    
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_zlim(global_min, global_max)
    ax.set_box_aspect([1, 1, 1])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    


def export_mesh(verts, faces, file_name="bunny_reconstructed128.ply"):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

   #mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(file_name,mesh)
