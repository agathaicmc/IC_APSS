import numpy as np
from skimage import measure
import open3d as o3d

def process_scalar_field(points, scalar_field, file_name,samples=25):



    try:

        verts, faces, _, _ = measure.marching_cubes(scalar_field, level=0.0)

        vector_min = np.min(points, axis=0)
        vector_max = np.max(points, axis=0)

        spacing = (vector_max - vector_min) / (samples - 1)
        verts = vector_min + verts * spacing
        

        export_mesh(verts, faces, file_name)

    except ValueError:
        print("Warning: no zero level surface found")    

    


def export_mesh(verts, faces, file_name):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    file_name = file_name + ".ply"
   #mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(file_name,mesh)
