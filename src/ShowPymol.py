from pymol import cmd
import numpy as np
import tempfile
import os
from rdkit import Chem
from pymol.cgo import *
from MolecularFieldCalculator import CoMSIAField 

def visualize_field_isosurface(data, field_type="electrostatic", top_percent=0.5, bottom_percent=0.5, transparency=0.7, name="field_isosurface"):
    """
    使用 Marching Cubes 算法生成分子场等值面并可视化到 PyMOL
    
    参数:
      data: 可为两种数据格式之一：
            1. numpy 数组，形状为 (N,4)，前三列为 x, y, z，第四列为 field_value。
            2. List[Tuple(x, y, z, field_value)]
      field_type: 场类型，可选: electrostatic/hydrophobic/steric/hbond_acceptor/hbond_donor，默认 "electrostatic"
      top_percent: 高值百分比阈值 (0-1)，默认 0.5
      bottom_percent: 低值百分比阈值 (0-1)，默认 0.5
      transparency: 透明度 (0.0-1.0)，默认 0.7
      name: PyMOL 对象名称，默认 "field_isosurface"
    """
    import pyvista as pv

    # 处理输入数据：两种格式均能提取出点坐标和对应的场值
    if isinstance(data, np.ndarray):
        if data.shape[1] != 4:
            print("错误: numpy 数组的形状应为 (N, 4) [x, y, z, field_value]")
            return
        points = data[:, :3]
        values = data[:, 3]
    elif isinstance(data, list):
        try:
            points = np.array([[x, y, z] for (x, y, z, v) in data])
            values = np.array([v for (x, y, z, v) in data])
        except Exception as e:
            print("错误: 列表数据格式异常，期望 List[Tuple(x, y, z, field_value)]")
            return
    else:
        print("错误: 数据格式不支持，必须为 numpy 数组 或 List[Tuple(x, y, z, field_value)]")
        return

    # 创建点云并设置场值属性
    point_cloud = pv.PolyData(points)
    point_cloud["values"] = values

    # 计算阈值：低、高阈值分别取场值的相应百分位
    high_threshold = np.percentile(values, 100 * (1 - top_percent))
    low_threshold = np.percentile(values, 100 * bottom_percent)

    # 使用 Delaunay 生成网格
    mesh = point_cloud.delaunay_3d()

    # 生成等值面 (移除了 method 参数以避免 vtkMarchingCubes 要求 vtkImageData 的问题)
    contours = mesh.contour([low_threshold, high_threshold], scalars="values")

    if contours.n_points == 0:
        print("警告: 未生成任何等值面，请检查输入数据和阈值设置")
        return

    # 转换为 PyMOL CGO 对象
    vertices = contours.points
    faces = contours.faces.reshape(-1, 4)[:, 1:4]

    obj = []
    # 定义颜色映射
    color_map = {
        'electrostatic': {'low': [0, 0, 1], 'high': [1, 0, 0]},         # blue, red
        'hydrophobic':  {'low': [0.5, 0, 0.5], 'high': [1, 0.5, 0]},       # purple, orange
        'steric':       {'low': [0, 1, 0], 'high': [1, 1, 0]},             # green, yellow
        'hbond_acceptor': {'low': [0.5, 0, 0.5], 'high': [0, 0.5, 0.5]},     # purple, teal
        'hbond_donor':    {'low': [0.5, 0.5, 0.5], 'high': [1, 1, 0]}        # grey, yellow
    }

    if field_type in color_map:
        # 添加低值等值面
        obj.extend([COLOR, *color_map[field_type]['low'],
                    ALPHA, transparency,
                    BEGIN, TRIANGLES])
        for face in faces:
            for idx in face:
                v = vertices[idx]
                obj.extend([VERTEX, v[0], v[1], v[2]])
        obj.append(END)
        # 添加高值等值面
        obj.extend([COLOR, *color_map[field_type]['high'],
                    ALPHA, transparency,
                    BEGIN, TRIANGLES])
        for face in faces:
            for idx in face:
                v = vertices[idx]
                obj.extend([VERTEX, v[0], v[1], v[2]])
        obj.append(END)

    # 加载 CGO 对象到 PyMOL 并设置显示属性
    cmd.load_cgo(obj, name)
    cmd.set("surface_mode", 1, name)
    cmd.set("transparency", transparency, name)
    cmd.zoom(name, 5)

def convert_comsia_to_visualization_data(comsia_field, field_key="steric"):
    """
    将 CoMSIAField 对象中的 grid_nodes 数据转换为可视化使用的数据格式。
    
    参数:
      comsia_field: CoMSIAField 对象，其中包含 grid_nodes 列表，每个节点具有 center 属性和各类场值。
      field_key: 指定需要使用的场类型，支持的值包括:
                 "steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor"
    
    返回:
      (numpy_data, list_data)
        numpy_data: numpy 数组，形状为 (N, 4)，每行依次为 [x, y, z, field_value]
        list_data:  list 格式，每个元素为 (x, y, z, field_value)
    """
    data_list = []
    
    for node in comsia_field.grid_nodes:
        # 获取节点中心坐标（此处假定 node.center 为一个懒加载属性或方法返回节点中心坐标元组）
        x, y, z = node.center
        
        # 根据 field_key 选择对应的场值
        if field_key == "steric":
            value = node.steric_value
        elif field_key == "electrostatic":
            value = node.electrostatic_value
        elif field_key == "hydrophobic":
            value = node.hydrophobic_value
        elif field_key == "hbond_donor":
            value = node.hbond_donor_value
        elif field_key == "hbond_acceptor":
            value = node.hbond_acceptor_value
        else:
            raise ValueError(f"不支持的 field_key: {field_key}. 请使用 'steric', 'electrostatic', 'hydrophobic', 'hbond_donor' 或 'hbond_acceptor'.")
        
        # 将节点数据添加到列表中
        data_list.append((x, y, z, value))
    
    # 同时生成 numpy 数组格式的数据
    numpy_data = np.array(data_list, dtype=float)
    
    return numpy_data, data_list


def visualize_all_fields(filename, file_type="hdf5", method="numpy", top_percent="0.5", bottom_percent="0.5", transparency="0.7"):
    """
    从指定的 HDF5 或 pickle 文件加载 CoMSIAField 对象，
    并从中提取五种场（steric, electrostatic, hydrophobic, hbond_donor, hbond_acceptor）的数据，
    同时从保存的 RDKit binary 数据直接恢复分子对象，通过 SDF 格式加载到 PyMOL，
    最后将所有生成的对象归纳到同一个 PyMOL group 内，便于统一管理。
    
    参数:
      filename: 存储 CoMSIAField 对象的文件名，例如 "molecule_comsia.h5" 或 "molecule_comsia.pkl"
      file_type: 文件类型，应为 "hdf5" 或 "pickle"，默认 "hdf5"
      method: 数据格式选择，"numpy" 或 "list"，决定转换后传给 visualize_field_isosurface 的数据格式；默认 "numpy"
      top_percent: 高值百分比阈值（0~1），默认 "0.5"
      bottom_percent: 低值百分比阈值（0~1），默认 "0.5"
      transparency: 透明度（0.0 ~ 1.0），默认 "0.7"
    """
    # 将字符串参数转换为 float
    top_percent = float(top_percent)
    bottom_percent = float(bottom_percent)
    transparency = float(transparency)
    
    # 根据 file_type 加载 CoMSIAField 对象
    if file_type.lower() == "hdf5":
        field_obj = CoMSIAField.load_from_hdf5(filename)
    elif file_type.lower() == "pickle":
        field_obj = CoMSIAField.load_from_pickle(filename)
    else:
        print("错误: 文件类型必须为 'hdf5' 或 'pickle'")
        return
    
    # 创建 group 用于归纳所有对象
    group_name = "all_fields_group"
    cmd.delete(group_name)  # 删除同名组（如果已存在）
    
    # 1. 加载分子结构：从存储的 RDKit binary 数据重构 Mol 对象
    try:
        mol = Chem.Mol(field_obj.serialized_mol.binary)
        # 将 Mol 对象转换为 SDF 格式字符串（SDF格式中每个分子后跟 "$$$$"）
        sdf_block = Chem.MolToMolBlock(mol)
        
        # 写入临时文件；注意 delete=False 使文件存在一段时间，以便加载后再删除
        with tempfile.NamedTemporaryFile(mode='w', suffix=".sdf", delete=False) as tmp_file:
            tmp_file.write(sdf_block)
            tmp_filename = tmp_file.name
        
        # 加载 SDF 文件到 PyMOL（对象名称为 "molecule_from_sdf"）
        cmd.load(tmp_filename, "molecule_from_sdf")
        os.remove(tmp_filename)  # 删除临时文件
        print("成功通过 SDF加载分子结构，生成对象名称：molecule_from_sdf")
        # 将分子对象加入 group 中
        # cmd.group(group_name, "molecule_from_sdf")
    except Exception as e:
        print("加载分子结构失败：", e)
    
    # 2. 展示各个场的等值面
    field_keys = ["steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor"]
    for field_key in field_keys:
        # 利用辅助函数 extract 数据：
        # convert_comsia_to_visualization_data 接收 CoMSIAField 对象与 field_key，
        # 返回两种格式的数据：numpy_data (N,4) 和列表格式。
        numpy_data, list_data = convert_comsia_to_visualization_data(field_obj, field_key=field_key)
        data = numpy_data if method.lower() == "numpy" else list_data
        obj_name = f"{field_key}"
        visualize_field_isosurface(data, field_type=field_key, top_percent=top_percent, bottom_percent=bottom_percent, transparency=transparency, name=obj_name)
        print(f"已可视化 {field_key} 场数据，生成对象名称：{obj_name}")
        # 将场数据对象加入 group
        cmd.group(group_name, members=obj_name)
    
    cmd.zoom("all")
    print(f"所有对象已归纳到 group: {group_name}")
    
# 注册为 PyMOL 命令，使用时可在 PyMOL 命令行输入，例如：
#   visualize_all_fields molecule_comsia.h5, hdf5, numpy, 0.5, 0.5, 0.7
cmd.extend("visualize_all_fields", visualize_all_fields)
cmd.extend("visualize_field_isosurface", visualize_field_isosurface)

## test command
'''
cd /srv/project/qsarmlkit/qsarmlkit/pycomsia/src/
load /srv/project/qsarmlkit/data/molecule_files/align_data/A-10.sdf
run /srv/project/qsarmlkit/qsarmlkit/pycomsia/src/ShowPymol.py
visualize_all_fields /srv/project/qsarmlkit/data/h5/A-10_fields.h5, hdf5, numpy, 0.5, 0.5, 0.7
解释：

molecule_comsia.h5: 是你保存的 HDF5 文件名。

hdf5: 指定使用 HDF5 格式加载（或可选 "pickle"）。

numpy: 使用 numpy 数组格式数据传递给渲染函数（也可使用 "list"）。

0.5, 0.5, 0.7: 分别对应 top_percent、bottom_percent 以及透明度参数。
'''
