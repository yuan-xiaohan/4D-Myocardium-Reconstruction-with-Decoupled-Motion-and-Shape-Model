import numpy as np
from mesh_to_sdf.mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere, scale_to_unit_cube,\
    sample_sdf_near_surface, sample_inside_surface, ComputeNormalizationParameters, transformation
import trimesh
import os
from deep_sdf.obj_process import obj_read


def writeSDFToNPZ(filename, points, sdf):
    pos_xyz = points[np.where(sdf > 0)]
    pos_sdf = sdf[np.where(sdf > 0)].reshape(-1, 1)
    pos = np.concatenate((pos_xyz, pos_sdf), axis=1)

    neg_xyz = points[np.where(sdf < 0)]
    neg_sdf = sdf[np.where(sdf < 0)].reshape(-1, 1)
    neg = np.concatenate((neg_xyz, neg_sdf), axis=1)

    np.savez(filename, pos=pos, neg=neg)

    """load npz file"""
    # sdf_load = np.load(filename)
    # print(sdf_load)


def PreprocessMesh(input_name, output_name, normalization_param_filename, test_sampling=True, num_sample=30000):
    mesh = trimesh.load(input_name)
    mesh = scale_to_unit_sphere(mesh)
    sampling_type = "sphere"
    # Sample some points uniformly and some points near the shape surface and calculate SDFs for these points.
    # This follows the procedure proposed in the DeepSDF paper. The mesh is first transformed to fit inside the unit sphere.
    points, sdf = sample_sdf_near_surface(mesh, sampling_type=sampling_type, test_sampling=test_sampling)
    writeSDFToNPZ(output_name, points, sdf)

    # Get Normalization Parameters and export npz file
    surface_points = mesh.sample(num_sample)
    offset, scale = ComputeNormalizationParameters(surface_points)
    np.savez(normalization_param_filename, offset=offset, scale=scale)


def SampleVisibleMeshSurface(input_name, output_name, normalization_param_filename, num_sample=30000):
    mesh = trimesh.load(input_name)

    # Sample points on the mesh surface and export ply file
    # surface_points = sample_inside_surface(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)
    surface_points = mesh.sample(num_sample)
    mesh_out = trimesh.Trimesh(vertices=surface_points)
    mesh_out.export(output_name)

    # Get Normalization Parameters and export npz file
    offset, scale = ComputeNormalizationParameters(surface_points)
    np.savez(normalization_param_filename, offset=offset, scale=scale)


def transform_to_canonical(base_path, instance, output_path, test_sampling=False):
    """
    Args:
        base_path
        instance: input mesh name
        output_path
        test_sampling: if it is test
    Return :
        processed: xyz, sdf, t
        Normalizarion parameters
    """
    print("Process patient: " + instance)
    out_patient_dir = os.path.join(output_path, instance)
    if not os.path.isdir(out_patient_dir):
        os.makedirs(out_patient_dir)
    obj_list = os.listdir(os.path.join(base_path, "points"))
    input_points_path = os.path.join(base_path, "points")

    # Get ED: phase = 00
    vertices_ED, _ = obj_read(os.path.join(input_points_path, "01.obj"))

    T = np.loadtxt(os.path.join(base_path, 'P.txt'))
    Ti = np.identity(4)
    Ti[0:3, 0:3] = np.linalg.inv(T[0:3, 0:3])
    Ti[0:3, 3] = -np.dot(np.linalg.inv(T[0:3, 0:3]), T[0:3, 3])

    # Get normalization parameters
    shape_lvv_1 = transformation(T, vertices_ED.transpose())
    offset, scale = ComputeNormalizationParameters(shape_lvv_1)
    # shape_lvv_2 = (shape_lvv_1 + offset) * scale  # transform to canonical space

    for i in range(len(obj_list)):  # i is phase
        phase = os.path.splitext(obj_list[i])[0]
        print("        Phase: " + phase)
        output_path = os.path.join(out_patient_dir, "%02d" % i + ".npz")
        # #### recover to origin space
        # v_3 = v_2 / scale - offset
        # homogeneous = np.column_stack((v_3, np.ones([v_3.shape[0], 1])))
        # v_4 = np.dot(Ti, homogeneous.transpose())[0:3, :].transpose()
        if not test_sampling:
            # For training
            phase_obj = os.path.join(base_path, "mesh", phase + ".obj")  # input mesh
            v_in, f = obj_read(phase_obj)
            v_1 = transformation(T, v_in.transpose())
            v_2 = (v_1 + offset) * scale
            mesh = trimesh.Trimesh(vertices=v_2, faces=f - 1)
            points, sdf = sample_sdf_near_surface(mesh,
                                                  sampling_type="sphere",
                                                  test_sampling=test_sampling)
            # colors = np.zeros(points.shape)
            # colors[sdf < 0, 2] = 1
            # colors[sdf > 0, 0] = 1
            # cloud = pyrender.Mesh.from_points(points, colors=colors)
            # scene = pyrender.Scene()
            # scene.add(cloud)
            # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

            pos_xyz = points[np.where(sdf > 0)]
            pos_sdf = sdf[np.where(sdf > 0)].reshape(-1, 1)
            pos = np.concatenate((pos_xyz, pos_sdf), axis=1)
            neg_xyz = points[np.where(sdf < 0)]
            neg_sdf = sdf[np.where(sdf < 0)].reshape(-1, 1)
            neg = np.concatenate((neg_xyz, neg_sdf), axis=1)
            np.savez(output_path,
                     pos=pos,
                     neg=neg,
                     t=i/(len(obj_list)-1),
                     T=T,
                     Ti=Ti,
                     offset=offset,
                     scale=scale
                     )
        else:
            # For testing
            phase_obj = os.path.join(base_path, "points", phase + ".obj")  # input points mesh
            v_in, f = obj_read(phase_obj)
            v_1 = transformation(T, v_in.transpose())
            v_2 = (v_1 + offset) * scale

            pos_xyz = v_2
            pos_sdf = np.zeros([pos_xyz.shape[0], 1])
            pcd = np.concatenate((pos_xyz, pos_sdf), axis=1)
            np.savez(output_path,
                     pcd=pcd,
                     t=i/(len(obj_list)-1),
                     T=T,
                     Ti=Ti,
                     offset=offset,
                     scale=scale)
