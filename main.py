import time

import open3d.io
from memory_profiler import profile
from reconstruction_system import launch
from reconstruction_system.utils import visualize_point_cloud
from reconstruction_system.utils import plot_results
from reconstruction_system .reconstruction import Reconstruction
from reconstruction_system.cfg import Config
import open3d as o3d

def get_depth_and_color_frames_from_dir(dir, index):
    image_index = f'{index:05}'
    rgb = o3d.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
    depth = o3d.io.read_image(f'{dir}\\depth\\{image_index}.png')
    return rgb, depth

@profile
def main():
    image_index = 1
    limit = 1000
    config = Config()
    dataset_path = 'E:\\all_dataset'
    rgb, depth = get_depth_and_color_frames_from_dir(dataset_path, 0)
    reconstruction = Reconstruction(config, rgb, depth)
    kek = time.time()
    for _ in range(limit):
        rgb, depth = get_depth_and_color_frames_from_dir(dataset_path, image_index)
        reconstruction.launch(rgb, depth)
        image_index += 1
        print(image_index)
    print(f'FINAL TIME : {time.time() - kek}')
    reconstruction.visualize_pcd()


if __name__ == '__main__':
    main()
    #plot_results()
    #visualize_point_cloud(open3d.io.read_point_cloud('E:\\Open3DProject\\reconstruction_system\\temp\\1017\\scene\\optimized_result_pcd_01017.ply'))
