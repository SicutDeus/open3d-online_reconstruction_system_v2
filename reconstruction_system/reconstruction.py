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
        self.source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(initial_rgb, initial_depth),
            self.config.intrinsic,
            self.config.extrinsic,
        )
        self.result_pcd = copy.deepcopy(self.source_pcd)
        self.icp_success = True

    def __update_target_pcd_via_frames(self, rgb_frame, depth_frame):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_frame, depth_frame)
        self.target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.config.intrinsic, self.config.extrinsic)
        self.target_pcd_not_transformed = copy.deepcopy(self.target_pcd)

    def __register_pcds(self):
        transformation, information = self.__register_point_cloud_pair()
        if transformation is None:
            return None
        self.__update_posegraph_for_scene(transformation, information)
        self.__optimize_posegraph_for_scene()

    def __register_point_cloud_pair(self):
        self.source_pcd = self.__preprocess_point_cloud(self.source_pcd)
        self.target_pcd = self.__preprocess_point_cloud(self.target_pcd)
        transformation, information = self.__multiscale_icp()
        return transformation, information

    def __preprocess_point_cloud(self, pcd):
        pcd = pcd.voxel_down_sample(self.config.voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.voxel_size * 2.0,
                                                 max_nn=30))
        return pcd

    def __update_posegraph_for_scene(self, transformation, information):
        odometry_inv = np.linalg.inv(np.dot(transformation, self.config.odometry))
        self.source_posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
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
        self.source_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        self.source_pcd.remove_radius_outlier(nb_points=8, radius=0.1)
        self.source_pcd = self.source_pcd.voxel_down_sample(voxel_size=self.config.final_optimization_voxel_size)

    def __integrate_pcd_into_scene(self):
        self.target_pcd.transform(self.source_posegraph.nodes[-1].pose)
        self.result_pcd += self.target_pcd
        if self.config.should_be_filtered:
            self.__filter_result_point_cloud()

    def __multiscale_icp(self):
        try:
            current_transformation = self.config.init_transformation
            max_iter = [self.config.max_iter]
            voxel_size = [self.config.voxel_size]
            for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
                iter = max_iter[scale]
                distance_threshold = self.config.voxel_size * 1.4
                if self.config.icp_method == "point_to_point":
                    result_icp = o3d.pipelines.registration.registration_icp(
                        self.source_pcd, self.target_pcd, distance_threshold,
                        current_transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        ),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=iter))
                else:
                    self.source_pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                                    2.0,
                                                             max_nn=30))
                    self.target_pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                                    2.0,
                                                             max_nn=30))
                    if self.config.icp_method == "point_to_plane":
                        result_icp = o3d.pipelines.registration.registration_icp(
                            self.source_pcd, self.target_pcd, distance_threshold,
                            current_transformation,
                            o3d.pipelines.registration.
                            TransformationEstimationPointToPlane(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                max_iteration=iter))
                    if self.config.icp_method == "color":
                        result_icp = o3d.pipelines.registration.registration_colored_icp(
                            self.source_pcd, self.target_pcd, voxel_size[scale],
                            current_transformation,
                            o3d.pipelines.registration.
                            TransformationEstimationForColoredICP(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                relative_fitness=1e-6,
                                relative_rmse=1e-6,
                                max_iteration=iter))
                    if self.config.icp_method == "generalized":
                        result_icp = o3d.pipelines.registration.registration_generalized_icp(
                            self.source_pcd, self.target_pcd, distance_threshold,
                            current_transformation,
                            o3d.pipelines.registration.
                            TransformationEstimationForGeneralizedICP(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                relative_fitness=1e-6,
                                relative_rmse=1e-6,
                                max_iteration=iter))
                current_transformation = result_icp.transformation
                if i == len(max_iter) - 1:
                    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        self.source_pcd, self.target_pcd, voxel_size[scale] * 1.4,
                        result_icp.transformation)
        except Exception as e:
            self.icp_success = False
        if self.icp_success:
            return result_icp.transformation, information_matrix
        return None, None

    def launch(self, rgb_frame, depth_frame):
        self.__update_target_pcd_via_frames(rgb_frame, depth_frame)
        self.__register_pcds()
        if self.icp_success:
            self.__integrate_pcd_into_scene()
            self.current_image_index += 1
            self.source_pcd = copy.deepcopy(self.target_pcd_not_transformed)
        self.icp_success = True


    def visualize_pcd(self):
        #vis_pcd = self.__rotate_pcd(self.source_pcd)
        vis_pcd = self.result_pcd
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

