from pymol import cmd
import numpy as np
from pymol.cgo import *

def visualize_selection_grid(selection_name="sele", grid_size=1.0, field_type="electrostatic", transparency=0.7):
    """
    可视化PyMOL选择对象的网格格子，使用PDB格式虚拟原子表示
    :param selection_name: 选择对象名称
    :param grid_size: 网格间距，默认1.0
    :param field_type: 场类型，默认为静电
    :param transparency: 透明度(0.0-1.0)，默认0.7
    :param top_percent: 高值百分比，默认0.5
    :param bottom_percent: 低值百分比，默认0.5
    """
    # 获取选择对象的坐标
    model = cmd.get_model(selection_name)
    if not model.atom:
        print(f"错误: 选择对象'{selection_name}'为空")
        return
        
    coords = np.array([atom.coord for atom in model.atom])
    
    # 计算网格参数
    grid_padding = 2.0
    min_coords = coords.min(axis=0) - grid_padding
    max_coords = coords.max(axis=0) + grid_padding
    
    # 生成网格点
    x = np.arange(min_coords[0], max_coords[0], grid_size)
    y = np.arange(min_coords[1], max_coords[1], grid_size)
    z = np.arange(min_coords[2], max_coords[2], grid_size)
    
    # 创建PDB格式字符串
    pdb_str = ""
    for i, (xi, yi, zi) in enumerate(np.ndindex(len(x), len(y), len(z))):
        x_coord = min_coords[0] + xi * grid_size
        y_coord = min_coords[1] + yi * grid_size
        z_coord = min_coords[2] + zi * grid_size
        
        # 使用bfactor存储场值（这里使用1.0作为示例值）
        bfactor = 1.0
        pdb_str += f"HETATM{i+1:5d}  X   GRD X   1    {x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00 {bfactor:6.2f}\n"
    
    # 加载到PyMOL
    cmd.read_pdbstr(pdb_str, f"{selection_name}_grid")
    
    # 设置显示样式
    cmd.show_as("spheres", f"{selection_name}_grid")
    cmd.set("sphere_scale", 0.3, f"{selection_name}_grid")
    # 确保透明度在0-1范围内
    transparency = max(0.0, min(1.0, transparency))
    cmd.set("sphere_transparency", transparency, f"{selection_name}_grid")
    # 确保透明对象可以被选中
    cmd.set("sphere_mode", 1, f"{selection_name}_grid")
    
    # 调整显示效果
    cmd.show("sticks", selection_name)
    cmd.zoom(selection_name, 5)
    
    # 解绑所有原子以显示场形状
    cmd.unbond(f"{selection_name}_grid", "*")
    
    # 设置颜色映射
    color_map = {
        'electrostatic': {'low': 'blue', 'high': 'red'},
        'hydrophobic': {'low': 'purple', 'high': 'orange'},
        'steric': {'low': 'green', 'high': 'yellow'},
        'hbond_acceptor': {'low': 'purple', 'high': 'teal'},
        'hbond_donor': {'low': 'grey', 'high': 'yellow'}
    }
    
    # 应用颜色映射
    if field_type in color_map:
        cmd.color(color_map[field_type]['low'], f"{selection_name}_grid and b < 0")
        cmd.color(color_map[field_type]['high'], f"{selection_name}_grid and b > 0")

def create_virtual_atom(coords, transparency=0.7, radius=0.5, bfactor=1.0, name="virtual_atom", ligand_id="GRD", chain_id="X"):
    """
    批量创建虚拟原子
    :param coords: 坐标数组或列表，格式为numpy数组或List[tuple(x,y,z)]
    :param transparency: 透明度(0.0-1.0)，默认0.7
    :param radius: 球体半径，默认0.5
    :param bfactor: bfactor值，默认1.0
    :param name: 对象名称，默认"virtual_atom"
    :param ligand_id: 配体ID，默认"GRD"
    :param chain_id: 链ID，默认"X"
    """
    # 创建PDB格式字符串
    pdb_str = ""
    if isinstance(coords, np.ndarray):
        coords = [tuple(coord) for coord in coords]
    
    for i, (x, y, z) in enumerate(coords):
        pdb_str += f"HETATM{i+1:5d}  X   {ligand_id} {chain_id}   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 {bfactor:6.2f}\n"
    
    # 加载到PyMOL
    cmd.read_pdbstr(pdb_str, name)
    
    # 设置显示样式
    cmd.show_as("spheres", name)
    cmd.set("sphere_scale", radius, name)
    transparency = max(0.0, min(1.0, transparency))
    cmd.set("sphere_transparency", transparency, name)
    cmd.set("sphere_mode", 1, name)
    cmd.zoom(name, 5)

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

def generate_test_field_data(selection="sele"):
    """
    从选择对象中提取原子坐标，并随机生成场值（范围 [-1,1]），生成两种数据格式：
      1. numpy 数组，形状为 (N, 4)：[x, y, z, field_value]
      2. tuple list: List[Tuple(x, y, z, field_value)]
    
    在 PyMOL 命令行打印出测试数据的示例信息，并返回字典：
      {'numpy': numpy_data, 'list': list_data}
    
    使用方法:
       PyMOL> generate_test_field_data sele
    """
    model = cmd.get_model(selection)
    if not model.atom:
        print(f"错误: 选择对象 '{selection}' 为空")
        return None

    coords_list = [atom.coord for atom in model.atom]
    np_coords = np.array(coords_list)
    # 随机生成场值
    np_field = np.random.uniform(-1.0, 1.0, size=len(np_coords))
    # 构造 numpy 数据：水平合并生成 (N, 4) 的数组
    numpy_data = np.hstack([np_coords, np_field.reshape(-1, 1)])
    # 构造 tuple list 数据
    list_data = [(x, y, z, f) for (x, y, z), f in zip(coords_list, np_field)]

    print("测试数据格式1 (numpy arrays):")
    print(f"  坐标数组形状: {np_coords.shape}")
    print(f"  场值数组示例: {np_field[:3]}")
    print("\n测试数据格式2 (tuple list):")
    print(f"  示例数据点: {list_data[0]}")
    
    return {'numpy': numpy_data, 'list': list_data}
# 新增：生成测试场数据的函数
def test_field_isosurface():
    """
    调用 generate_test_field_data 获取测试数据，并分别使用两种数据格式调用 visualize_field_isosurface 进行等值面渲染测试。
    
    使用方法:
       PyMOL> test_field_isosurface
    """
    data = generate_test_field_data()
    if data is None:
        return

    print("使用 numpy 数组格式数据进行等值面渲染测试...")
    visualize_field_isosurface(data['numpy'], field_type="electrostatic", top_percent=0.5, bottom_percent=0.5, transparency=0.7, name="field_isosurface_numpy")

    print("使用 tuple list 格式数据进行等值面渲染测试...")
    visualize_field_isosurface(data['list'], field_type="electrostatic", top_percent=0.5, bottom_percent=0.5, transparency=0.7, name="field_isosurface_list")
    cmd.zoom("all")

# 注册 PyMOL 命令
cmd.extend("visualize_field_isosurface", visualize_field_isosurface)
cmd.extend("generate_test_field_data", generate_test_field_data)
cmd.extend("test_field_isosurface", test_field_isosurface)
# 注册PyMOL命令
cmd.extend("visualize_grid", visualize_selection_grid)
cmd.extend("create_virtual_atom", create_virtual_atom)