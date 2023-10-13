## USE THIS FILE TO GENERATE NEW DEFORMATIONS FOR THE PAPER "Fast and simple statistical shape analysis of pregnant women
## using radial deformation of a cylindrical template"

from meshGraph import *
import numpy as np
import os
import pandas as pd
import time
import copy

# Gets average radial distance of eight neighboring vertices of cylinder mesh from the obj mesh
def getAverageNeighborDistance(vertexInd, distFromMeshFaceDict, theta):
    neighboring_distances = 0
    num_of_neighbors = 0
    avg_neighbor_distance = 0
    if (vertexInd - 1) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd - 1]
        num_of_neighbors += 1
    if (vertexInd + 1) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd + 1]
        num_of_neighbors += 1
    if (vertexInd - (360 / theta)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd - (360 / theta)]
        num_of_neighbors += 1
    if (vertexInd + (360 / theta)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd + (360 / theta)]
        num_of_neighbors += 1
    if (vertexInd - ((360 / theta) + 1)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd - ((360 / theta) + 1)]
        num_of_neighbors += 1
    if (vertexInd + ((360 / theta) + 1)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd + ((360 / theta) + 1)]
        num_of_neighbors += 1
    if (vertexInd - ((360 / theta) - 1)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd - ((360 / theta) - 1)]
        num_of_neighbors += 1
    if (vertexInd + ((360 / theta) - 1)) in distFromMeshFaceDict:
        neighboring_distances += distFromMeshFaceDict[vertexInd + ((360 / theta) - 1)]
        num_of_neighbors += 1
    if num_of_neighbors > 0:
        avg_neighbor_distance = neighboring_distances / num_of_neighbors
    return avg_neighbor_distance

def getDist(points, correspondences, weights):
    dist_squared = np.sum((points - correspondences)**2, axis=1)
    return np.sum(np.multiply(weights, dist_squared))

def objective(T, points, radius_vecs, corresponding_points, distance_weights, edges, optimization_weights):
    # Distance error
    new_points = []
    for i in range(0, len(T)):
        new_point = (T[i] * radius_vecs[i, :])
        new_points.append(points[i, :] + new_point)
    new_points = np.asarray(new_points)
    distance_error = getDist(new_points, corresponding_points, distance_weights)
    print(distance_error)
    # Smoothness error
    smoothness_error = 0
    for edge in edges:
        frobenius_norm_squared = ((T[edge.startNode] - T[edge.endNode]) ** 2) + \
                                     np.sum((radius_vecs[edge.startNode, :] - radius_vecs[edge.endNode, :]) ** 2)
        smoothness_error = smoothness_error + frobenius_norm_squared
    return (optimization_weights[0] * distance_error) + (optimization_weights[1] * smoothness_error)


# processed_df = pd.read_excel("Period_4_OBJs/Processed OBJs.xlsx", sheet_name="Sheet1")
processed_df = pd.read_excel("Variability (Inter)/Processed OBJs.xlsx", sheet_name="Sheet1")
# obj_folder = "Period_4_OBJs/Processed_OBJs"
obj_folder = "Variability (Inter)/Processed_OBJs"
obj_process_dictionary = {}
num_obj_processed = 0
for i in range(len(processed_df)):
    obj = processed_df.iloc[i, 0]
    # if num_obj_processed == 10:
    #     break
    # if obj != "01-03-0123-EBA-4.obj":
    #     continue
    # if obj not in {"01-03-0012-AGS-4.obj", "01-03-0041-BEC-4.obj", "01-04-0065-SWA-4.obj"}:
    #     continue
    if obj == ".DS_Store":
        continue
    # deformed_parameters_df = pd.read_excel("Period_4_OBJs/Deformation_Parameters_9/" + obj.split(".")[0] + ".xlsx",
    #                                        sheet_name="Deformation Parameters")
    # if len(deformed_parameters_df) == 9090:
    #     continue
    try:
        print(obj)
        # obj_process_dictionary[obj] = ["", "", "", "", "", "", ""]
        obj_process_dictionary[obj] = ["", "", "", "", "", ""]
        obj_split = obj.split(".")[0].split("-")
        obj_process_dictionary[obj][0] = obj_split[0]
        obj_process_dictionary[obj][1] = obj_split[1]
        obj_process_dictionary[obj][2] = obj_split[2]
        try:
            obj_process_dictionary[obj][3] = obj_split[3]
        except:
            obj_process_dictionary[obj][0] = obj_split[0]
            obj_process_dictionary[obj][1] = ""
            obj_process_dictionary[obj][2] = obj_split[1]
            obj_process_dictionary[obj][3] = obj_split[2]
        # obj_process_dictionary[obj][3] = obj_split[3]
        # obj_process_dictionary[obj][4] = obj_split[4]
        mg = meshGraph()
        mg.populateMeshGraph(obj_folder + "/" + obj)
        mg.computeSortedIndices()
        ## Normalize the mesh to a unit sphere
        mesh_centroid = mg.getMeshMean()
        mg.translate(mesh_centroid)
        mesh_vertices = np.asarray(mg.getVerticesList())
        max_distance = np.max(np.sqrt(np.sum(abs(mesh_vertices) ** 2, axis=1)))
        mg.scaleVertices(max_distance)
        # rotation_angle = float(processed_df.iloc[i, 8]) * (-1)
        rotation_angle = float(processed_df.iloc[i, 7]) * (-1)
        mg.rotateAboutAxis(rotation_angle, [0, 0, 0], Axis.y)
        mg.computeSortedIndices()
        mg.populatePCD()
        # mg.save_pointcloud("normalized_mesh.obj")
        # jkvj
        ## Generate template mesh
        # template_trimesh = trimesh.creation.icosphere(subdivisions=5, radius=1.01)
        # trimesh.exchange.export.export_mesh(template_trimesh, "template_sphere.obj")
        # kk
        template = meshGraph()
        degree_resolution = 4
        original_y_resolution = 0.02
        length = 2
        radius = 1
        y_sections = int((length/original_y_resolution) + 1)
        # y_sections = 128
        template.generateCylinder(0.0, 0.0, mg.getMinValue(Axis.y), mg.getMaxValue(Axis.y), y_sections, radius=radius,
                                  theta_resolution=degree_resolution)
        # print(len(template.vertices))
        # rotation_angle = float(processed_df.iloc[i, 8])
        # template.rotateAboutAxis(rotation_angle, [0, 0, 0], Axis.y)
        # template.save_pointcloud("template_cylinder.obj")
        # Get corresponding points of template mesh
        start_time = time.time()
        # template = meshGraph()
        # template.populateMeshGraph("template_cylinder.obj")
        non_deforming_vertex_indices = set()
        hole_vertex_indices = set()
        dist_from_face_dict = {}
        radius_vecs_dict = {}
        deformation_parameter_dict = {}
        for y_val in np.linspace(mg.getMinValue(Axis.y), mg.getMaxValue(Axis.y), num=y_sections):
            # print(y_val)
            y_radius_indices = mg.getVertexIndicesAround(y_val, Axis.y, radius=0.015)
            # if not y_radius_indices:
            #     non_deforming_vertex_indices.update(template.yCoordinateDictionary[y_val])
            #     for vertex_ind in template.yCoordinateDictionary[y_val]:
            #         vertex = template.vertices[vertex_ind]
            #         deformation_parameter_dict[vertex_ind] = [0, vertex.x, vertex.y, vertex.z]
            #     continue
            for vertex_ind in template.yCoordinateDictionary[y_val]:
                vertex = template.vertices[vertex_ind]
                a = 0 - vertex.z
                b = (0 - vertex.x) * (-1)
                c = ((a * vertex.x) + (b * vertex.z)) * (-1)
                radius_vec = [-b, 0, a]
                radius_vec = radius_vec / np.linalg.norm(radius_vec)
                radius_vecs_dict[vertex_ind] = radius_vec
                # Find deformation distance and deform the cylinder mesh vertex
                radial_face_indices = set()
                for ind in y_radius_indices:
                    point = [mg.vertices[ind].x, mg.vertices[ind].y, mg.vertices[ind].z]
                    dist_from_line = abs((a * point[0]) + (b * point[2]) + c) / math.sqrt(a ** 2 + b ** 2)
                    if dist_from_line <= 0.01:
                        radial_face_indices.update(mg.verticesFaceDictionary[ind])
                min_distance = 1000
                deformed_vertex = None
                face_intersection_index = None
                for face_index in radial_face_indices:
                    dist_from_face, _ = mg.intersectsFace([vertex.x, vertex.y, vertex.z], radius_vec,
                                                                       face_index, back_face_culling=True)
                    if dist_from_face:
                        if dist_from_face < min_distance:
                            min_distance = dist_from_face
                            new_vertex = [vertex.x + (dist_from_face * radius_vec[0]),
                                          vertex.y + (dist_from_face * radius_vec[1]),
                                          vertex.z + (dist_from_face * radius_vec[2])]
                            deformed_vertex = new_vertex
                            face_intersection_index = face_index
                if deformed_vertex:
                    dist_from_face_dict[vertex_ind] = min_distance
                    vertex.x = deformed_vertex[0]
                    vertex.z = deformed_vertex[2]
                    deformation_parameter_dict[vertex_ind] = [min_distance, vertex.x, vertex.y, vertex.z]
                else:
                    hole_vertex_indices.add(vertex_ind)
                    deformation_parameter_dict[vertex_ind] = []
        # template.save_pointcloud("temp.obj")
        # print("Saved")
        ## Fill holes by iteratively averaging
        while hole_vertex_indices:
            new_hole_vertex_indices = []
            for vertex_ind in hole_vertex_indices:
                avg_neighbor_distance = getAverageNeighborDistance(vertex_ind, dist_from_face_dict,
                                                                   degree_resolution)
                if avg_neighbor_distance > 0:
                    vertex = template.vertices[vertex_ind]
                    a = template.zCenter - vertex.z
                    b = (template.xCenter - vertex.x) * (-1)
                    c = ((a * vertex.x) + (b * vertex.z)) * (-1)
                    radius_vec = [-b, 0, a]
                    radius_vec = radius_vec / np.linalg.norm(radius_vec)
                    dist_from_face_dict[vertex_ind] = avg_neighbor_distance
                    new_vertex = [vertex.x + (avg_neighbor_distance * radius_vec[0]),
                                  vertex.y + (avg_neighbor_distance * radius_vec[1]),
                                  vertex.z + (avg_neighbor_distance * radius_vec[2])]
                    deformed_vertex = new_vertex
                    vertex.x = deformed_vertex[0]
                    vertex.z = deformed_vertex[2]
                    deformation_parameter_dict[vertex_ind] = [avg_neighbor_distance, vertex.x, vertex.y, vertex.z]
                else:
                    new_hole_vertex_indices.append(vertex_ind)
            hole_vertex_indices = copy.deepcopy(new_hole_vertex_indices)
        end_time = time.time()
        time_taken = end_time - start_time
        print(time_taken)
        # template.save_pointcloud("Period_4_OBJs/Deformed_OBJs_13/" + obj)
        template.save_pointcloud("Variability (Inter)/Deformed_OBJs_1/" + obj)
        obj_process_dictionary[obj][4] = time_taken
        deformation_parameters_df = pd.DataFrame.from_dict(deformation_parameter_dict, orient='index',
                                                                   columns=["Radial Deformation", "X", "Y", "Z"])
        deformation_settings_df = pd.DataFrame.from_dict({"Radius": radius, "Y Resolution": original_y_resolution,
                                                          "Theta Resolution": degree_resolution, "Length": length},
                                                         orient='index')
        # with pd.ExcelWriter("Period_4_OBJs/Deformation_Parameters_13/" + obj.split(".")[0] + ".xlsx", mode='w') as \
        #         writer:
        with pd.ExcelWriter("Variability (Inter)/Deformation_Parameters_1/" + obj.split(".")[0] + ".xlsx", mode='w') as \
                writer:
            deformation_parameters_df.to_excel(writer, sheet_name='Deformation Parameters')
            deformation_settings_df.to_excel(writer, sheet_name='Deformation Settings')
    except ImportError:
        print("ERRORED")
        obj_process_dictionary[obj][5] += "Errored, "


# processed_obj_df = pd.DataFrame.from_dict(obj_process_dictionary, orient='index', columns=["Site ID", "Region ID",
#                                                                                            "Subject ID", "Subject Ini",
#                                                                                            "Visit", "Time Taken",
#                                                                                            "Errored?"])
processed_obj_df = pd.DataFrame.from_dict(obj_process_dictionary, orient='index', columns=["Inter", "Nurse ID",
                                                                                           "Subject ID", "Camera ID",
                                                                                           "Time Taken", "Errored?"])
# processed_obj_df.to_excel("Period_4_OBJs/Deformation_OBJs_13.xlsx")
processed_obj_df.to_excel("Variability (Inter)/Deformation_OBJs_1.xlsx")