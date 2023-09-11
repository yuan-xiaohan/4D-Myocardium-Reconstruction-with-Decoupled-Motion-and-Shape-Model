import numpy as np


def obj_read(obj_path):
    # read obj file and get vertices and faces
    with open(obj_path) as file:
        vertices = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
            if strs[0] == "f":
                faces.append(
                    (int(strs[1].split('//')[0]), int(strs[2].split('//')[0]), int(strs[3].split('//')[0])))
    vertices = np.array(vertices)  # in matrix form
    faces = np.array(faces)
    return vertices, faces

# if have no face:obj_process.obj_write(name, vertives, np.zeros(0))
def obj_write(obj_path, vertices, faces=np.zeros([0])):
    with open(obj_path, "w") as file:
        file.write("# " + str(vertices.shape[0]) + " vertices, " + str(faces.shape[0]) + " faces" + "\n")
        # write vertices
        for i in range(vertices.shape[0]):
            file.write("v ")
            file.write(str(float("{0:.6g}".format(vertices[i, 0]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 1]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 2]))) + " ")
            file.write("\n")
        # write faces
        for i in range(faces.shape[0]):
            file.write("f ")
            file.write(str(int(faces[i, 0])) + " ")
            file.write(str(int(faces[i, 1])) + " ")
            file.write(str(int(faces[i, 2])) + " ")
            file.write("\n")
