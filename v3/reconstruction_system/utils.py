import open3d as o3d
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt
from reconstruction_system.config import config


def get_depth_and_color_frames_from_dir(dir, index):
    '''
    Получает ргб кадр и кадр глубины из директории
    :param dir: директория из которой берутся кадры
    :param index: индекс изображения
    :return: ргб кадр и кадр глубины
    '''
    image_index = f'{index:05}'
    rgb = o3d.io.read_image(f'{dir}\\rgb\\{image_index}.jpg')
    depth = o3d.io.read_image(f'{dir}\\depth\\{image_index}.png')
    return rgb, depth


def make_clean_folder(path_folder):
    '''
    Создает/очищает папку
    :param path_folder: папка, которую нужно очистить
    :return: None
    '''
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    else:
        shutil.rmtree(path_folder)
        os.mkdir(path_folder)


def clear_previous_temp_folder(image_index):
    '''
    Очищает предыдущие папки внутри temp за ненадобностью
    :param image_index: индекс изображения, от которого отсчитывается предыдущая папка
    :return: None
    '''
    shutil.rmtree(f'{config["temp_dir"]}\\{image_index - 2}')


def make_path_into_temp_dir(inner_path, image_index=-1):
    '''
    Создаёт путь внутри папки temp
    :param inner_path: путь, внутри папки temp
    :param image_index: индекс(название папки) к которой необходимо получить путь
    :return: путь до нужной папки
    '''
    if inner_path.count(config['default_filling']) > 0:
        return os.path.join(config['temp_dir'] + f'\\{image_index}',
                            inner_path % image_index)
    return os.path.join(config['temp_dir'] + f'\\{image_index}', inner_path)


def write_info_about_frame(image_index, pcd, rgb_frame, depth_frame):
    '''
    Записывает в директорию информацию о кадре
    :param image_index: индекс(название) папки
    :param pcd: поинт клауд
    :param rgb_frame: ргб кадр
    :param depth_frame: кадр глубингы
    :return: None
    '''
    o3d.io.write_point_cloud(make_path_into_temp_dir(config['default_frame_point_cloud_name'], image_index), pcd)
    o3d.io.write_image(make_path_into_temp_dir(config['default_rgb_frame_name'], image_index), rgb_frame)
    o3d.io.write_image(make_path_into_temp_dir(config['default_depth_frame_name'], image_index), depth_frame)


def clear_log_files():
    '''
    Очищает/Создаёт файлы для локального логирования
    :return: None
    '''
    open(config['weights_log_filename'], 'w').close()
    open(config['times_log_filename'], 'w').close()
    open(config['keypoints_log_filename'], 'w').close()

def calculate_execution_time(func):
    '''
    Считает время выполнения функции
    :param func: оборачиваемая функция
    :return: время выполнения функции
    '''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        return f'{round(time.time() - start_time,3)}'
    return wrapper


def write_pcd_size(path):
    '''
    Считает и записывает в лог файл вес поинт клауда
    :param path:
    :return:
    '''
    with open(config['weights_log_filename'], 'a') as f:
        f.write(f'{round(os.path.getsize(path) / 1024, 3)} КБ\n')


'''
    Отображение информации из лог файлов
'''

def plot_timestamps(timestamps, name):
    '''
    Отображает переданные на вход отметки
    :param timestamps: временные/количественные отметки
    :param name: название создаваемого окна
    :return: None
    '''
    plt.figure(num=name)
    xstamps = [i for i in range(0, len(timestamps))]
    plt.plot(xstamps, timestamps, 'b', label='line one', linewidth=5)
    plt.show()


def plot_results():
    '''
    Отображает в виде графиков данные из локальных лог файлов
    :return: None
    '''
    weights_stamps = []
    proccess_timestamps = []
    register_timestamps = []
    integrate_timestamps = []
    keypoints_stamps = []
    with open(config['weights_log_filename'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            weights_stamps.append(line.split(' ')[0])
        plot_timestamps(weights_stamps, 'weight of point cloud')
    with open(config['keypoints_log_filename'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            keypoints_stamps.append(line)
        plot_timestamps(keypoints_stamps, 'Number of keypoints')
    with open('times_log_filename', 'r') as f:
        lines = f.readlines()
        for line in lines:
            splited = line.split(' ')
            proccess_timestamps.append(splited[0])
            register_timestamps.append(splited[1])
            integrate_timestamps.append(splited[2])
        plot_timestamps(proccess_timestamps, 'proccess frame')
        plot_timestamps(register_timestamps, 'register frames')
        plot_timestamps(integrate_timestamps, 'integrate frames')




'''
    Визуализация поинт клаудов
'''
def add_grid_on_vis(vis, pcd):
    '''
    Добавляет к окну визуализации сетку,построенную на основе поинт клауда
    :param vis: окно визуализации
    :param pcd: поинт клауд
    :return: None
    '''
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
            lines.colors = o3d.utility.Vector3dVector(config['red_color'])  # Red color
            vis.add_geometry(lines)

    # Create grid lines along the Y-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for z in np.arange(min_bound[2], max_bound[2], voxel_size):
            points = np.array([[x, min_bound[1], z], [x, max_bound[1], z]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector(config['green_color'])
            vis.add_geometry(lines)

    # Create grid lines along the Z-axis
    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for y in np.arange(min_bound[1], max_bound[1], voxel_size):
            points = np.array([[x, y, min_bound[2]], [x, y, max_bound[2]]])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            lines.colors = o3d.utility.Vector3dVector(config['blue_color'])
            vis.add_geometry(lines)


def rotate_pcd(pcd):
    '''
    Переворачивает поинт клауд для корректной визуализации
    :param pcd: поинт клауд
    :return: None
    '''
    axis = (1, 0, 0)
    angle = np.pi
    axis = np.array(axis)
    R = pcd.get_rotation_matrix_from_axis_angle(axis * angle)
    pcd.rotate(R, center=(0, 0, 0))

def visualize_point_cloud(pcd, grid=False):
    '''
    Визуализирование поинт клауда
    :param pcd: поинт клауда
    :param grid: нужна ли сетка
    :return: None
    '''
    rotate_pcd(pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    if grid:
        add_grid_on_vis(vis, pcd)
    vis.run()