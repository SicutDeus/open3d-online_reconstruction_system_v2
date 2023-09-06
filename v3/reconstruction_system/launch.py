import os
from reconstruction_system import process_single_rgbd_image
from reconstruction_system import utils
from reconstruction_system.config import config
from reconstruction_system import register_frames
from reconstruction_system import integrate_scene



def add_frame_to_pcd(initial_pcd, rgb_frame, depth_frame, image_index):
    pcd = process_single_rgbd_image.run(rgb_frame, depth_frame, image_index)
    pose_graph = register_frames.run(image_index)
    integrate_scene.run()
    return None

def reconstruct(limit=-1, is_from_dir=True, should_visualize=True, rgb_frame=None, depth_frame=None, image_index=0):
    '''
    Запуск онлайн реконструкции для кадров поступающих с камеры / взятых из директории
    :param limit: ограничение по количеству кадров, взятых из директории
    :param is_from_dir: нужно ли брать из директории или получать от камеры
    :param should_visualize: нужно ли визулизировать облако точек
    :param rgb_frame: ргб кадр
    :param depth_frame: кадр глубины
    :param image_index: индекс изображения
    :return: None
    '''
    utils.make_clean_folder(config['temp_dir'])
    utils.clear_log_files()
    if is_from_dir:
        rgb_frame, depth_frame = utils.get_depth_and_color_frames_from_dir(config['images_dir'], image_index + 1)
        initial_pcd = process_single_rgbd_image.run(rgb_frame, depth_frame)
        for _ in os.listdir(f'{config["images_dir"]}\\rgb'):
            rgb_frame, depth_frame = utils.get_depth_and_color_frames_from_dir(config['images_dir'], image_index+1)

            success = add_frame_to_pcd(initial_pcd, rgb_frame, depth_frame)

            image_index += 1
            if image_index == limit:
                break

    else:
        success = add_frame_to_pcd(rgb_frame, depth_frame, image_index)


    if should_visualize:
        utils.visualize_point_cloud(config['optimized_result_pcd'])