import open3d as o3d
import numpy as np
class Config:
    def __init__(self):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        self.extrinsic = np.eye(4)
        self.odometry = np.identity(4)
        self.init_transformation = np.identity(4)
        self.downsample_voxel_size = 0.01
        self.voxel_size = 0.05
        self.final_optimization_voxel_size = 0.2
        self.depth_diff_max = 0.07
        self.max_correspondence_distance = 0.05 * 1.4
        self.preference_loop_closure = 5.0
        self.edge_prune_threshold = 0.25
        self.tsdf_cubic_size = 3.0
        self.max_iter = 50
        self.reference_node = 0

        self.global_registration = 'none'
        self.opencv_slam = False

        self.images_dir = 'E:\\all_dataset'

        self.result_pcd = o3d.geometry.PointCloud()
        self.optimized_result_pcd = o3d.geometry.PointCloud()
        self.posegraph_edges = 5
        self.icp_method = 'color'
        self.odometry_case = True

        self.should_be_filtered = False

        self.grid = False

        self.red_color = [[1, 0, 0]]
        self.green_color = [[0, 1, 0]]
        self.blue_color = [[0, 0, 1]]