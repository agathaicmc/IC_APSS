import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

def plot_surface(points, ax, scalar_field, color, file_name,samples=25):



    try:

        verts, faces, _, _ = measure.marching_cubes(scalar_field, level=0.0)

        vector_min = np.min(points, axis=0)
        vector_max = np.max(points, axis=0)

        spacing = (vector_max - vector_min) / (samples - 1)
        verts = vector_min + verts * spacing
        
        mesh = Poly3DCollection(verts[faces], alpha=0.8, edgecolor='none')
        mesh.set_facecolor(color)
        mesh.set_edgecolor('black')
        mesh.set_linewidth(0.2)
        ax.add_collection3d(mesh)

        export_mesh(verts, faces, file_name)

    except ValueError:
        print("Warning: no zero level surface found")    
    
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_zlim(global_min, global_max)
    ax.set_box_aspect([1, 1, 1])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    


def export_mesh(verts, faces, file_name):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    file_name = file_name + ".ply"
   #mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(file_name,mesh)
