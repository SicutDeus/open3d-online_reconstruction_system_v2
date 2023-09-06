import copy
import sys
import time

import numpy as np

from reconstruction_system.cfg import Config
import open3d as o3d


class Reconstruction:
    def __init__(self, config: Config, initial_rgb, initial_depth):
        self.source_posegraph = config.pose_graph
        self.config = config
        self.target_pcd = None
        self.current_image_index = 1
        self.source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(initial_rgb, initial_depth)
        self.source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.source_rgbd, self.config.intrinsic, self.config.extrinsic)
        self.current_rgbd = None
        self.result_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(initial_rgb, initial_depth),
            self.config.intrinsic,
            self.config.extrinsic,
        )
        self.icp_success = True
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config.tsdf_cubic_size / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def __update_target_pcd_via_frames(self, rgb_frame, depth_frame):
        self.current_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_frame, depth=depth_frame, convert_rgb_to_intensity=False)
        self.target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.current_rgbd, intrinsic=self.config.intrinsic, extrinsic=self.config.extrinsic)
    def __register_pcds(self):
        transformation, information = self.__register_point_cloud_pair()
        if transformation is None:
            return None
        self.__update_posegraph_for_scene(transformation, information)
        self.__optimize_posegraph_for_scene()

    def __register_point_cloud_pair(self):
        if self.current_image_index == 1:
            self.source_pcd, self.source_fpfh = self.__preprocess_point_cloud(self.source_pcd)
        self.target_pcd, self.target_fpfh = self.__preprocess_point_cloud(self.target_pcd)
        self.target_pcd_not_transformed = copy.deepcopy(self.target_pcd)
        if self.config.global_registration == 'ransac':
            return self.register_point_cloud_fpfh(self.source_pcd, self.target_pcd, self.source_fpfh, self.target_fpfh)
        return self.__multiscale_icp()

    def __preprocess_point_cloud(self, pcd):
        pcd = pcd.voxel_down_sample(self.config.voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.voxel_size * 2.0, max_nn=100))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.voxel_size * 5.0, max_nn=100))
        return pcd, pcd_fpfh

    def __update_posegraph_for_scene(self, transformation, information):
        self.source_posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(transformation))
        self.source_posegraph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(self.current_image_index - 1,
                                                     self.current_image_index,
                                                     transformation,
                                                     information,
                                                     uncertain=False))
        if self.current_image_index > self.config.posegraph_edges:
            self.source_posegraph = self.cut_posegraphv2()

    def __optimize_posegraph_for_scene(self):
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            self.config.max_correspondence_distance,
            self.config.edge_prune_threshold,
            self.config.preference_loop_closure,
            self.config.reference_node,
        )
        o3d.pipelines.registration.global_optimization(self.source_posegraph, method, criteria, option)

    def __filter_result_point_cloud(self):
        self.result_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        self.result_pcd.remove_radius_outlier(nb_points=8, radius=0.1)
        self.result_pcd = self.result_pcd.voxel_down_sample(voxel_size=self.config.final_optimization_voxel_size)

    def __integrate_pcd_into_scene(self):
        self.target_pcd.transform(self.source_posegraph.nodes[-1].pose)
        self.result_pcd += self.target_pcd
        if self.config.should_be_filtered:
            self.__filter_result_point_cloud()

    def __multiscale_icp(self):
        try:
            current_transformation = self.config.init_transformation
            distance_threshold = self.config.voxel_size * 0.5
            if self.config.icp_method == "point_to_plane":
                result_icp = o3d.pipelines.registration.registration_icp(
                    self.source_pcd, self.target_pcd, distance_threshold,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.config.max_iter))
            if self.config.icp_method == "color":
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    self.source_pcd, self.target_pcd, self.config.voxel_size,
                    current_transformation,
                    o3d.pipelines.registration.
                    TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=self.config.max_iter))
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    self.source_pcd, self.target_pcd, self.config.voxel_size * 1.4,
                    result_icp.transformation)
        except Exception as e:
            self.icp_success = False
        if self.icp_success:
            self.config.init_transformation = result_icp.transformation
            return result_icp.transformation, information_matrix
        return None, None

    def __integrate_rgbd_into_volume(self):
        self.volume.integrate(self.current_rgbd, self.config.intrinsic, np.linalg.inv(self.source_posegraph.nodes[-1].pose))


    def launch(self, rgb_frame, depth_frame):
        self.__update_target_pcd_via_frames(rgb_frame, depth_frame)
        self.__register_pcds()
        if self.icp_success:
            self.__integrate_pcd_into_scene()
            #self.__integrate_rgbd_into_volume()
            self.current_image_index += 1
            self.source_pcd = copy.deepcopy(self.target_pcd_not_transformed)
            self.source_rgbd = self.current_rgbd
        self.icp_success = True

    def visualize_volume(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = self.__rotate_pcd(mesh)
        vis.add_geometry(mesh)
        vis.run()

    def visualize_pcd(self):
        vis_pcd = self.__rotate_pcd(self.result_pcd)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(vis_pcd)
        if self.config.grid:
            self.__add_grid_on_vis(vis, vis_pcd)
        vis.run()

    def __add_grid_on_vis(self, vis, pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        voxel_size = 0.5
        for y in np.arange(min_bound[1], max_bound[1], voxel_size):
            for z in np.arange(min_bound[2], max_bound[2], voxel_size):
                points = np.array([[min_bound[0], y, z], [max_bound[0], y, z]])
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(points)
                lines.lines = o3d.utility.Vector2iVector([[0, 1]])
                lines.colors = o3d.utility.Vector3dVector(self.config.red_color)
                vis.add_geometry(lines)

        # Create grid lines along the Y-axis
        for x in np.arange(min_bound[0], max_bound[0], voxel_size):
            for z in np.arange(min_bound[2], max_bound[2], voxel_size):
                points = np.array([[x, min_bound[1], z], [x, max_bound[1], z]])
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(points)
                lines.lines = o3d.utility.Vector2iVector([[0, 1]])
                lines.colors = o3d.utility.Vector3dVector(self.config.green_color)
                vis.add_geometry(lines)

        # Create grid lines along the Z-axis
        for x in np.arange(min_bound[0], max_bound[0], voxel_size):
            for y in np.arange(min_bound[1], max_bound[1], voxel_size):
                points = np.array([[x, y, min_bound[2]], [x, y, max_bound[2]]])
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(points)
                lines.lines = o3d.utility.Vector2iVector([[0, 1]])
                lines.colors = o3d.utility.Vector3dVector(self.config.blue_color)
                vis.add_geometry(lines)

    def __rotate_pcd(self, pcd):
        axis = (1, 0, 0)
        angle = np.pi
        axis = np.array(axis)
        R = self.source_pcd.get_rotation_matrix_from_axis_angle(axis * angle)
        pcd.rotate(R, center=(0, 0, 0))
        return pcd

    def cut_posegraphv2(self):
        temp_posegraph = o3d.pipelines.registration.PoseGraph()
        source_edge_id = 0
        for id in range(1, len(self.source_posegraph.edges)):
            temp_posegraph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(self.source_posegraph.nodes[id])
            )
            temp_posegraph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_edge_id,
                    source_edge_id + 1,
                    self.source_posegraph.edges[id].transformation,
                    self.source_posegraph.edges[id].information
                )
            )
            source_edge_id += 1
        temp_posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(self.source_posegraph.nodes[len(self.source_posegraph.edges)]))
        return temp_posegraph

    def register_point_cloud_fpfh(self, source, target, source_fpfh, target_fpfh):
        distance_threshold = self.config.voxel_size * 1.4
        if self.config.global_registration == "fgr":
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=distance_threshold))
        if self.config.global_registration == "ransac":
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh, False, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False), 4,
                [
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100, 0.9))
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, distance_threshold, result.transformation)
        return result.transformation, information

    def slam_opencv(self):
        from reconstruction_system.opencv_pose_estimation import pose_estimation
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = self.config.depth_diff_max
        success_5pt, odo_init = pose_estimation(self.source_rgbd,
                                                self.current_rgbd,
                                                self.config.intrinsic, False)
        [success, trans, info
         ] = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.source_rgbd, self.current_rgbd, self.config.intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option)
        return trans, info

    def test_user_reg(self):
        trans_init = np.identity(4)

        # Set the probabilistic ICP parameters
        # You can adjust these values according to your data
        max_correspondence_distance = 0.05  # Maximum distance threshold for point correspondences
        criterion = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)  # Convergence criteria
        covariance = 0.01 * np.identity(6)  # Covariance matrix of the Gaussian noise

        # Perform probabilistic ICP
        reg_p2p = o3d.pipelines.registration.registration_icp_probabilistic(
            self.source_pcd, self.target_pcd, max_correspondence_distance, trans_init, criterion, covariance)