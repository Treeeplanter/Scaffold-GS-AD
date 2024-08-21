# This script is based on an original implementation by True Price.
# Created by liminghao
import sys
import sqlite3
import numpy as np
from glob import glob
import os
import shutil
from PIL import Image
from plyfile import PlyData, PlyElement
import open3d as o3d
from tqdm import tqdm


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2 ** 31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)



def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid
    
    def update_image(self, name,camera_id,prior_q,prior_t,image_id):
        cursor = self.execute(
            "UPDATE images SET name=?, camera_id=? WHERE image_id=?",
            (name, camera_id,image_id))
        return cursor.lastrowid
    # def update_image(self, name,camera_id,prior_q,prior_t,image_id):
    #     cursor = self.execute(
    #         "UPDATE images SET name=?, camera_id=?, prior_qw=?,prior_qx=?,prior_qy=?,prior_qz=?,prior_tx=?,prior_ty=?,prior_tz=? WHERE image_id=?",
    #         (name, camera_id, prior_q[0], prior_q[1], prior_q[2], prior_q[3],prior_t[0], prior_t[1],prior_t[2],image_id))
    #     return cursor.lastrowid

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        prior_q=np.full(4, np.NaN),
        prior_t=np.full(3, np.NaN),
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os
    import argparse

    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()

def imgTodatabase(txtfile, database_path):

    import os
    import argparse

    if os.path.exists(database_path)==False:
        print("ERROR: database path dosen't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    camList=list()
    qwList = list()
    qxList = list()
    qyList = list()
    qzList = list()
    txList = list()
    tyList = list()
    tzList = list()

    # Update real cameras from .txt
    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),2):
            prior_q=np.full(4, np.NaN)
            prior_t=np.full(3, np.NaN)
            prior_q = list(prior_q)
            prior_t = list(prior_t)
            if lines[i][0]!='#':
                strLists = lines[i].split()
                image_id = int(strLists[0])
                name = str(strLists[-1])
                camera_id = int(strLists[-2])
                prior_q[0] = float(strLists[1])
                prior_q[1] = float(strLists[2])
                prior_q[2] = float(strLists[3])
                prior_q[3] = float(strLists[4])
                prior_t[0] = float(strLists[5])
                prior_t[1] = float(strLists[6])
                prior_t[2] = float(strLists[7])

                idList.append(image_id)
                camList.append(camera_id)
                qwList.append(prior_q[0])
                qxList.append(prior_q[1])
                qyList.append(prior_q[2])
                qzList.append(prior_q[3])
                txList.append(prior_t[0])
                tyList.append(prior_t[1])
                tzList.append(prior_t[2])


                image_id = db.update_image(name,
                                        camera_id,
                                        prior_q,
                                        prior_t,
                                        image_id)

    # Commit the data to the file.
    db.commit()

    # Read and check cameras.
    # rows = db.execute("SELECT * FROM cameras")
    # for i in range(0,len(idList),1):
    #     iamge_id,name,camera_id,qw,qx,qy,qz,tx,ty,tz  = next(rows)
    #     params = blob_to_array(params, np.float64)
    #     assert image_id == idList[i]
    #     assert camera_id == int(1) and qw == qwList[i] 
       

    # Close database.db.
    db.close()

def get_table_columns(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    conn.close()
    return columns

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def extractCOLMAP_waymo(data_root, world_origin, strat_time,end_time,reas_freq):

    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], 
        [-1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, 0, 1]]
    )

    scene_root = data_root
    cache_dir = os.path.join(scene_root, "colmap_ws")
    cache_img_dir = os.path.join(cache_dir, "images")
    cache_sparse_dir = os.path.join(cache_dir, 'created','sparse')
    cache_triangulate_dir = os.path.join(cache_dir, 'triangulate','sparse','model')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cache_sparse_dir, exist_ok=True)
    os.makedirs(cache_triangulate_dir, exist_ok=True)
    os.makedirs(cache_img_dir, exist_ok=True)

    # save cameras.txt
    # load intrinsics
    load_size = [640, 960]
    ORIGINAL_SIZE = [1280, 1920]
    intrinsic = np.loadtxt(os.path.join(scene_root, "intrinsics", "0.txt"))
    x, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]f
    # scale intrinsics w.r.t. load size
    # fx, fy = (
    #     fx * load_size[1] / ORIGINAL_SIZE[1],
    #     fy * load_size[0] / ORIGINAL_SIZE[0],
    # )
    # cx, cy = (
    #     cx * load_size[1] / ORIGINAL_SIZE[1],
    #     cy * load_size[0] / ORIGINAL_SIZE[0],
    # )
    f = open(os.path.join(cache_sparse_dir, "cameras.txt"), "a")
    
    f.write(f"1 PINHOLE {ORIGINAL_SIZE[1]} {ORIGINAL_SIZE[0]} {fx} {fy} {cx} {cy}\n")
    f.close()

    ego_to_world_start = world_origin
    for frame in range(0,198,1):
       
        img = os.path.join(scene_root, "images", f"{frame:03d}_0.jpg")
        if not os.path.exists(img):
            print(f"{img} doesn't exist!")
            continue
        # save image file to cache
        save_img_name =f"{frame:03d}_0.jpg"
        shutil.copyfile(img, os.path.join(cache_img_dir, save_img_name))

    
        ego_to_world_current = np.loadtxt(os.path.join(scene_root, "ego_pose", f"{frame:03d}.txt"))

        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current

        cam_to_ego = np.loadtxt(os.path.join(scene_root, "extrinsics", "0.txt")) @ OPENCV2DATASET 
        # transformation:
        # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
        c2w = ego_to_world @ cam_to_ego

        # save images.txt
        w2c = np.linalg.inv(c2w)
        R = rotmat2qvec(w2c[:3, :3])
        T = w2c[:3, 3].squeeze()
        f = open(os.path.join(cache_sparse_dir, "images.txt"), "a")
        f.write(f"{frame+1} {R[0]} {R[1]} {R[2]} {R[3]} {T[0]} {T[1]} {T[2]} 1 {save_img_name}\n")
        f.write("\n")
        f.close()
        
    
    # # create a blank points3d.txt file
    file = open(os.path.join(cache_sparse_dir, "points3D.txt"), "w")
    file.close() 

    # # Start COLMAP
    print("Starting COLMAP..")
    database_path = os.path.join(cache_dir, "database.db")


    # # Feature extraction          
    os.system(f"colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path {database_path} --image_path {cache_img_dir}")

    # #     # 使用示例
    db_path = database_path
    table_name = 'images'
    columns = get_table_columns(db_path, table_name)

    for col in columns:
        print(col)
    
    camTodatabase(os.path.join(cache_sparse_dir, "cameras.txt"),database_path)
    imgTodatabase(os.path.join(cache_sparse_dir, "images.txt"),database_path)
    
    # Feature matching
    os.system(f"colmap exhaustive_matcher --database_path {database_path}")

    # Triangulation with known camera parameters
    cache_triangulate_dir
    os.makedirs(cache_triangulate_dir, exist_ok=True)


    os.system(f"colmap point_triangulator --database_path {database_path} --image_path {cache_img_dir} --input_path {cache_sparse_dir} --output_path {cache_triangulate_dir}")
    
    # Convert to PLY 

    os.system(f"colmap model_converter --input_path {cache_triangulate_dir} --output_path {cache_triangulate_dir} --output_type TXT")
    os.system(f"colmap model_converter --input_path {cache_triangulate_dir} --output_path {cache_triangulate_dir}/points3D.ply --output_type PLY")

    
    process_dir = os.path.join(scene_root,"colmap_processed" )
    os.makedirs(process_dir, exist_ok=True)
    
    plydata = PlyData.read(cache_triangulate_dir + "/points3D.ply")

    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    
    # initialize points with open3d object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
    xyz = np.asarray(pcd.points)
    rgb = rgb[ind]

    # save ply
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([xyz, rgb], axis=1)))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])

    ply_data.write(process_dir + "/points3D.ply")
    # remove cache
    shutil.rmtree(cache_dir)
    return pcd.points
    