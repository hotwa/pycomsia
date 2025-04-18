from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from functools import cached_property
from joblib import Parallel, delayed
import h5py,pickle
import json

# 尝试多种方式导入SerializedMolecule
try:
    # 第一种尝试：直接从utils导入
    from utils import SerializedMolecule
except ImportError:
    try:
        # 第二种尝试：从当前目录下的utils模块导入
        from .utils import SerializedMolecule
    except ImportError:
        try:
            # 第三种尝试：从src.utils导入
            from src.utils import SerializedMolecule
        except ImportError:
            try:
                # 第四种尝试：从pycomsia.utils导入
                from pycomsia.utils import SerializedMolecule
            except ImportError as e:
                raise ImportError(
                    "无法导入SerializedMolecule。尝试了以下导入路径:\n"
                    "1. from utils import SerializedMolecule\n"
                    "2. from .utils import SerializedMolecule\n"
                    "3. from src.utils import SerializedMolecule\n"
                    "4. from pycomsia.utils import SerializedMolecule\n"
                    f"最终错误: {str(e)}"
                )

@dataclass
class GridNode:
    """表示3D网格中的一个节点"""
    # 网格索引
    i: int
    j: int
    k: int
    # 网格原点和间距
    grid_origin: Tuple[float, float, float]
    grid_spacing: Tuple[float, float, float]
    # 场值
    steric_value: float = 0.0
    electrostatic_value: float = 0.0
    hydrophobic_value: float = 0.0
    hbond_donor_value: float = 0.0
    hbond_acceptor_value: float = 0.0
    
    @cached_property
    def center(self) -> Tuple[float, float, float]:
        """惰性计算中心点坐标"""
        x0, y0, z0 = self.grid_origin
        dx, dy, dz = self.grid_spacing
        return (x0 + self.i * dx, y0 + self.j * dy, z0 + self.k * dz)
    
    @cached_property
    def vertices(self) -> List[Tuple[float, float, float]]:
        """惰性计算八个顶点坐标"""
        center_x, center_y, center_z = self.center
        dx, dy, dz = self.grid_spacing
        half_dx, half_dy, half_dz = dx/2, dy/2, dz/2
        
        return [
            (center_x - half_dx, center_y - half_dy, center_z - half_dz),  # 左下前
            (center_x + half_dx, center_y - half_dy, center_z - half_dz),  # 右下前
            (center_x - half_dx, center_y + half_dy, center_z - half_dz),  # 左上前
            (center_x + half_dx, center_y + half_dy, center_z - half_dz),  # 右上前
            (center_x - half_dx, center_y - half_dy, center_z + half_dz),  # 左下后
            (center_x + half_dx, center_y - half_dy, center_z + half_dz),  # 右下后
            (center_x - half_dx, center_y + half_dy, center_z + half_dz),  # 左上后
            (center_x + half_dx, center_y + half_dy, center_z + half_dz),  # 右上后
        ]


@dataclass
class MolecularField:
    """分子场的基类"""
    serialized_mol: SerializedMolecule
    mol_id: Optional[str] = None
    grid_nodes: List[GridNode] = field(default_factory=list)
    grid_dimensions: Tuple[int, int, int] = field(default_factory=tuple)
    grid_spacing: Tuple[float, float, float] = field(default_factory=tuple)
    grid_origin: Tuple[float, float, float] = field(default_factory=tuple)

    def __repr__(self) -> str:
        return (f"<MolecularField (mol_id={self.mol_id}, "
                f"serialized_mol={self.serialized_mol}, "
                f"grid_dimensions={self.grid_dimensions}, "
                f"grid_spacing={self.grid_spacing}, "
                f"grid_origin={self.grid_origin})>")
    
    def to_numpy(self) -> np.ndarray:
        """将场数据转换为3D numpy数组"""
        if not self.grid_nodes or not self.grid_dimensions:
            return np.array([])
        
        nx, ny, nz = self.grid_dimensions
        field_array = np.zeros((nx, ny, nz))
        
        for node in self.grid_nodes:
            # 直接使用节点的索引
            i, j, k = node.i, node.j, node.k
            
            # 确保索引在有效范围内
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                field_array[i, j, k] = self.get_value(node)
                
        return field_array

    def to_fields_dict(self) -> Dict[str, np.ndarray]:
        """
        根据当前 grid_nodes 和 grid_dimensions 返回一个字典，
        键为 'steric_field', 'electrostatic_field', 'hydrophobic_field',
        'hbond_donor_field', 'hbond_acceptor_field'，值为对应的 3D numpy 数组。
        """
        if not self.grid_nodes or not self.grid_dimensions:
            return {}
        
        nx, ny, nz = self.grid_dimensions
        steric = np.zeros((nx, ny, nz))
        electrostatic = np.zeros((nx, ny, nz))
        hydrophobic = np.zeros((nx, ny, nz))
        hbond_donor = np.zeros((nx, ny, nz))
        hbond_acceptor = np.zeros((nx, ny, nz))
        
        for node in self.grid_nodes:
            i, j, k = node.i, node.j, node.k
            steric[i, j, k] = node.steric_value
            electrostatic[i, j, k] = node.electrostatic_value
            hydrophobic[i, j, k] = node.hydrophobic_value
            hbond_donor[i, j, k] = node.hbond_donor_value
            hbond_acceptor[i, j, k] = node.hbond_acceptor_value
            
        return {
            'steric_field': steric,
            'electrostatic_field': electrostatic,
            'hydrophobic_field': hydrophobic,
            'hbond_donor_field': hbond_donor,
            'hbond_acceptor_field': hbond_acceptor
        }

    
    def get_value(self, node: GridNode) -> float:
        """获取节点的场值，由子类实现"""
        raise NotImplementedError("子类必须实现此方法")

@dataclass
class CoMSIAField(MolecularField):
    def __repr__(self) -> str:
        # 直接访问 molecule 的 serialized_mol 属性
        return (f"<CoMSIAField (name={self.serialized_mol.name}, "
                f"grid_dimensions={self.grid_dimensions}, "
                f"grid_spacing={self.grid_spacing}, "
                f"grid_origin={self.grid_origin})>")
    
    def get_value(self, node: GridNode) -> float:
        return node.steric_value

    def save_to_pickle(self, filename: str):
        """保存整个 CoMSIAField 对象到 pickle 文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"CoMSIAField 已保存到 {filename}")

    @classmethod
    def load_from_pickle(cls, filename: str) -> "CoMSIAField":
        """通过 pickle 从文件中加载 CoMSIAField 对象"""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f"CoMSIAField 已从 {filename} 加载")
        return obj

    @classmethod
    def load_from_pickle(cls, filename: str) -> "CoMSIAField":
        """通过 pickle 从文件中加载 CoMSIAField 对象"""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f"CoMSIAField 已从 {filename} 加载")
        return obj

    def save_to_hdf5(self, filename: str):
        """
        使用 HDF5 格式保存 CoMSIAField 数据。
        除了网格数据和各节点场值，还保存 serialized_mol 中的分子信息：
            - 二进制信息 binary（使用 np.void 保存）
            - properties（转换成 JSON 字符串）
            - name（分子名称）
        """
        coords = []
        steric_values = []
        electrostatic_values = []
        hydrophobic_values = []
        hbond_donor_values = []
        hbond_acceptor_values = []

        for node in self.grid_nodes:
            coords.append(node.center)
            steric_values.append(node.steric_value)
            electrostatic_values.append(node.electrostatic_value)
            hydrophobic_values.append(node.hydrophobic_value)
            hbond_donor_values.append(node.hbond_donor_value)
            hbond_acceptor_values.append(node.hbond_acceptor_value)

        coords = np.array(coords)
        steric_values = np.array(steric_values)
        electrostatic_values = np.array(electrostatic_values)
        hydrophobic_values = np.array(hydrophobic_values)
        hbond_donor_values = np.array(hbond_donor_values)
        hbond_acceptor_values = np.array(hbond_acceptor_values)

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("grid_dimensions", data=np.array(self.grid_dimensions))
            hf.create_dataset("grid_spacing", data=np.array(self.grid_spacing))
            hf.create_dataset("grid_origin", data=np.array(self.grid_origin))
            # 保存原有的 mol_id
            hf.attrs["mol_id"] = self.mol_id if self.mol_id is not None else ""
            # 保存 serialized_mol 里的内容
            hf.attrs["serialized_mol_name"] = self.serialized_mol.name
            hf.attrs["serialized_mol_binary"] = np.void(self.serialized_mol.binary)
            hf.attrs["serialized_mol_properties"] = json.dumps(self.serialized_mol.properties)
            hf.create_dataset("coords", data=coords)
            hf.create_dataset("steric_values", data=steric_values)
            hf.create_dataset("electrostatic_values", data=electrostatic_values)
            hf.create_dataset("hydrophobic_values", data=hydrophobic_values)
            hf.create_dataset("hbond_donor_values", data=hbond_donor_values)
            hf.create_dataset("hbond_acceptor_values", data=hbond_acceptor_values)
        print(f"CoMSIAField 数据已保存到 HDF5 文件: {filename}")

    @classmethod
    def load_from_hdf5(cls, filename: str) -> "CoMSIAField":
        """
        从 HDF5 文件中加载数据，并还原为 CoMSIAField 对象。
        包括网格数据、各节点场值以及 serialized_mol 中保存的分子信息。
        """
        with h5py.File(filename, 'r') as hf:
            grid_dimensions = tuple(hf["grid_dimensions"][()])
            grid_spacing = tuple(hf["grid_spacing"][()])
            grid_origin = tuple(hf["grid_origin"][()])
            mol_id = hf.attrs.get("mol_id", "")
            if isinstance(mol_id, bytes):
                mol_id = mol_id.decode('utf-8')
            if mol_id == "":
                mol_id = None

            serialized_mol_name = hf.attrs.get("serialized_mol_name", "")
            if isinstance(serialized_mol_name, bytes):
                serialized_mol_name = serialized_mol_name.decode('utf-8')
            # 注意：这里 serialized_mol_name 为空时，我们不做特别处理，因为 serialized_mol 已为必填项
            serialized_mol_binary_attr = hf.attrs.get("serialized_mol_binary", b"")
            if isinstance(serialized_mol_binary_attr, (bytes, np.void)):
                serialized_mol_binary = bytes(serialized_mol_binary_attr)
            else:
                serialized_mol_binary = b""

            serialized_mol_properties_str = hf.attrs.get("serialized_mol_properties", "{}")
            if isinstance(serialized_mol_properties_str, bytes):
                serialized_mol_properties_str = serialized_mol_properties_str.decode('utf-8')
            serialized_mol_properties = json.loads(serialized_mol_properties_str)

            coords = hf["coords"][()]
            steric_values = hf["steric_values"][()]
            electrostatic_values = hf["electrostatic_values"][()]
            hydrophobic_values = hf["hydrophobic_values"][()]
            hbond_donor_values = hf["hbond_donor_values"][()]
            hbond_acceptor_values = hf["hbond_acceptor_values"][()]

        # 构造 serialized_mol 字典（必填项）
        serialized_mol = SerializedMolecule(
            binary=serialized_mol_binary,
            name=serialized_mol_name,
            properties=serialized_mol_properties
        )

        field_obj = cls(
            grid_dimensions=grid_dimensions,
            grid_spacing=grid_spacing,
            grid_origin=grid_origin,
            serialized_mol=serialized_mol  # 必传入
        )
        field_obj.mol_id = mol_id

        field_obj.grid_nodes = []
        nx, ny, nz = grid_dimensions
        index = 0
        total_nodes = nx * ny * nz
        if total_nodes != len(coords):
            print("警告: 读取的节点总数与 grid_dimensions 不匹配！")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if index < len(coords):
                        node = GridNode(
                            i=i,
                            j=j,
                            k=k,
                            grid_origin=grid_origin,
                            grid_spacing=grid_spacing,
                            steric_value=float(steric_values[index]),
                            electrostatic_value=float(electrostatic_values[index]),
                            hydrophobic_value=float(hydrophobic_values[index]),
                            hbond_donor_value=float(hbond_donor_values[index]),
                            hbond_acceptor_value=float(hbond_acceptor_values[index])
                        )
                        field_obj.grid_nodes.append(node)
                        index += 1
        print(f"CoMSIAField 已从 HDF5 文件 {filename} 加载")
        return field_obj

class MolecularFieldCalculator:
    def __init__(self):
        self.ALPHA = 0.3
        self.TOLERANCE = 1e-4
        
        self.donor_smarts = [
            #"[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"
            "[$([#7,#8,#15,#16]);H]"
        ]
        
        self.acceptor_smarts = [
            # "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$("
            # "[N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
            "[#8&!$(*~N~[OD1]),#7&H0;!$([D4]);!$([D3]-\*=,:[$([#7,#8,#15,#16])])]"
        ]
        
        self.donor_patterns = [Chem.MolFromSmarts(smart) for smart in self.donor_smarts]
        self.acceptor_patterns = [Chem.MolFromSmarts(smart) for smart in self.acceptor_smarts]

    def _process_mol(self, mol_binary: SerializedMolecule, is_training, grid_spacing, grid_dimensions, grid_origin):
        """
        包装单个分子的场计算，返回 (fields, structured_fields, is_training)
        """
        fields, structured_fields = self._calc_single_molecule_field(mol_binary, grid_spacing, grid_dimensions, grid_origin)
        return fields, structured_fields, is_training
        
    def calc_field(self, 
               aligned_results: List[Tuple[Mol, bool]],
               grid_spacing: Tuple[float, float, float],
               grid_dimensions: Tuple[int, int, int],
               grid_origin: Tuple[float, float, float],
               return_structured: bool = False) -> dict:
        """
        Calculate molecular fields for both training and prediction sets.
        
        Args:
            aligned_results: 对齐后的分子结果列表，每项为 (mol, is_training) 元组
            grid_spacing: 网格间距
            grid_dimensions: 网格维度
            grid_origin: 网格原点
            return_structured: 是否返回结构化场数据，默认为 False
            
        Returns:
            如果 return_structured 为 False，返回原始场数据字典；否则返回包含原始场和结构化场数据的字典。
        """
        train_fields = {
            'steric_field': [],
            'electrostatic_field': [],
            'hydrophobic_field': [],
            'hbond_donor_field': [],
            'hbond_acceptor_field': []
        }
        
        pred_fields = {
            'steric_field': [],
            'electrostatic_field': [],
            'hydrophobic_field': [],
            'hbond_donor_field': [],
            'hbond_acceptor_field': []
        }
        
        # 结构化场数据统一为列表，每个元素是一个 CoMSIAField 对象
        train_structured_fields = []
        pred_structured_fields = []
        
        # 并行处理所有分子
        results = Parallel(n_jobs=-1)(
        delayed(self._process_mol)(
            SerializedMolecule.from_mol(mol),  # 序列化分子为二进制字符串
            is_training,
            grid_spacing,
            grid_dimensions,
            grid_origin
        )
        for mol, is_training in aligned_results if mol is not None
    )
        
        # 根据 is_training 标记分别存储计算结果
        for fields, structured_fields, is_training in results:
            if is_training:
                for field_name in train_fields:
                    train_fields[field_name].append(fields[field_name].flatten())
                if return_structured:
                    train_structured_fields.append(structured_fields)
            else:
                for field_name in pred_fields:
                    pred_fields[field_name].append(fields[field_name].flatten())
                if return_structured:
                    pred_structured_fields.append(structured_fields)
        
        # 根据参数决定返回的数据结构
        if return_structured:
            return {
                'train_fields': train_fields,
                'pred_fields': pred_fields,
                'train_structured_fields': train_structured_fields,
                'pred_structured_fields': pred_structured_fields
            }
        else:
            return {
                'train_fields': train_fields,
                'pred_fields': pred_fields
            }
    
    @staticmethod
    def _update_node_field(node, steric_field, electrostatic_field, hydrophobic_field, hbond_donor_field, hbond_acceptor_field):
        """更新单个节点的场值"""
        i, j, k = node.i, node.j, node.k
        node.steric_value = steric_field[i, j, k]
        node.electrostatic_value = electrostatic_field[i, j, k]
        node.hydrophobic_value = hydrophobic_field[i, j, k]
        node.hbond_donor_value = hbond_donor_field[i, j, k]
        node.hbond_acceptor_value = hbond_acceptor_field[i, j, k]
        return node
    def _calc_single_molecule_field(self, mol_binary: SerializedMolecule, grid_spacing, grid_dimensions, grid_origin):
        dx, dy, dz = grid_spacing
        nx, ny, nz = grid_dimensions
        x0, y0, z0 = grid_origin

        # 初始化 numpy 数组来保存场数据
        steric_field = np.zeros((nx, ny, nz))
        electrostatic_field = np.zeros((nx, ny, nz))
        hydrophobic_field = np.zeros((nx, ny, nz))
        hbond_donor_field = np.zeros((nx, ny, nz))
        hbond_acceptor_field = np.zeros((nx, ny, nz))
        
        # 反序列化 Molecule 对象
        mol = mol_binary.to_mol()
        # 从分子中获取 mol_id（属性名按实际情况调整，此处使用 mol_id）
        mol_id = mol.GetProp('mol_id') if mol.HasProp('mol_id') else f"unknown"
        
        # 初始化统一结构化场数据
        comsia_field = CoMSIAField(grid_dimensions=grid_dimensions, 
                                grid_spacing=grid_spacing, 
                                grid_origin=grid_origin,
                                serialized_mol=mol_binary,
                                mol_id=mol_id)
        
        # 创建网格节点（仍然使用三重循环创建）
        grid_nodes = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    node = GridNode(
                        i=i, j=j, k=k,
                        grid_origin=grid_origin,
                        grid_spacing=grid_spacing
                    )
                    grid_nodes.append(node)
        
        # 获取分子的原子信息
        conformer = mol.GetConformer()
        atom_coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        
        # 计算 Gasteiger 电荷
        AllChem.ComputeGasteigerCharges(mol)
        atom_charges = np.array([float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()])
        atom_charges = np.nan_to_num(atom_charges)
        
        # 计算疏水性
        atom_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        atom_hydrophobicities = np.array([contrib[0] for contrib in atom_contribs])
        
        # 获取范德华半径
        ptable = Chem.GetPeriodicTable()
        atom_vdw = np.array([ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
        
        # 生成网格点，用于后续 numpy 运算
        x = x0 + np.arange(nx) * dx
        y = y0 + np.arange(ny) * dy
        z = z0 + np.arange(nz) * dz
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
        
        # 计算每个原子的贡献（使用 numpy 数组运算）
        for i in range(mol.GetNumAtoms()):
            displacement = grid_points - atom_coords[i]
            distances_sq = np.sum(displacement ** 2, axis=1)
            gaussian = np.exp(-self.ALPHA * distances_sq)
            steric_field.ravel()[:] -= (atom_vdw[i] ** 3) * gaussian
            electrostatic_field.ravel()[:] += (atom_charges[i] * gaussian) * 337 
            hydrophobic_field.ravel()[:] += (atom_hydrophobicities[i] * gaussian) * 25
        
        # 计算 H-键相关场（donor 和 acceptor）
        for pattern in self.donor_patterns:
            matches = mol.GetSubstructMatches(pattern, uniquify = 1)
            pseudoatoms = self.generate_pseudoatoms(mol, matches, True)
            for pos in pseudoatoms:
                displacement = grid_points - pos
                distances_sq = np.sum(displacement ** 2, axis=1)
                gaussian = np.exp(-self.ALPHA * distances_sq)
                hbond_donor_field.ravel()[:] += gaussian * 20
        
        for pattern in self.acceptor_patterns:
            matches = mol.GetSubstructMatches(pattern, uniquify = 1)
            pseudoatoms = self.generate_pseudoatoms(mol, matches, False)
            for pos in pseudoatoms:
                displacement = grid_points - pos
                distances_sq = np.sum(displacement ** 2, axis=1)
                gaussian = np.exp(-self.ALPHA * distances_sq)
                hbond_acceptor_field.ravel()[:] += gaussian * 20
        
        # 并行更新所有网格节点的场值
        grid_nodes = Parallel(n_jobs=-1)(
            delayed(MolecularFieldCalculator._update_node_field)(node, steric_field, electrostatic_field, hydrophobic_field, 
                                        hbond_donor_field, hbond_acceptor_field)
            for node in grid_nodes
        )
        
        # 将更新后的节点赋值给统一的结构化场对象
        comsia_field.grid_nodes = grid_nodes.copy()
        
        # 原始场数据以字典形式返回
        fields = {
            'steric_field': steric_field,
            'electrostatic_field': electrostatic_field,
            'hydrophobic_field': hydrophobic_field,
            'hbond_donor_field': hbond_donor_field,
            'hbond_acceptor_field': hbond_acceptor_field
        }
        
        # 统一结构化场数据直接使用 comsia_field 对象返回
        structured_fields = comsia_field
        
        return fields, structured_fields
    
    def generate_pseudoatoms(self, mol, matches, is_donor):
        """Generate pseudoatom positions for H-bond donors/acceptors."""
        pseudoatom_positions = []
        
        for match in matches:
            atom_idx = match[0]
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_pos = np.array(mol.GetConformer().GetAtomPosition(atom_idx))
            positions = self._get_hbond_positions(mol, atom, atom_pos, is_donor)
            
            # Pass the donor/acceptor atom index
            filtered_positions = self._filter_positions(mol, positions, is_donor, donor_acceptor_idx=atom_idx)
            
            if not filtered_positions and is_donor:
                print(f"Warning: All donor positions filtered out for atom {atom_idx}")
                
            pseudoatom_positions.extend(filtered_positions)
        
        return pseudoatom_positions
    
    def _get_hbond_positions(self, mol, atom, atom_pos, is_donor):
        """Generate positions for hydrogen bond interactions."""
        positions = []
        distance = 1.9  # Ångstroms
        
        if is_donor:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    h_pos = np.array(mol.GetConformer().GetAtomPosition(neighbor.GetIdx()))
                    vector = h_pos - atom_pos
                    vector /= np.linalg.norm(vector)
                    positions.append(atom_pos + distance * vector)
        else:
            vectors = self._get_hybridization_vectors(atom)
            for vector in vectors:
                positions.append(atom_pos + distance * vector)
        return positions
    
    def _get_hybridization_vectors(self, atom):
        """Get vectors based on atom hybridization."""
        vectors = []
        hybridization = atom.GetHybridization()
        
        if hybridization == Chem.rdchem.HybridizationType.SP2:
            vectors = [
                np.array([1, 0, 0]),
                np.array([-0.5, np.sqrt(3)/2, 0]),
                np.array([-0.5, -np.sqrt(3)/2, 0])
            ]
        elif hybridization == Chem.rdchem.HybridizationType.SP3:
            vectors = [
                np.array([1, 1, 1]),
                np.array([-1, -1, 1]),
                np.array([-1, 1, -1]),
                np.array([1, -1, -1])
            ]
        else:
            vectors = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1])
            ]
        
        return [vec / np.linalg.norm(vec) for vec in vectors]
    
    def _filter_positions(self, mol, positions, is_donor, donor_acceptor_idx=None):
        """
        Remove positions that are too close to any non-hydrogen atom in the molecule,
        excluding the donor/acceptor atom itself.
        
        Args:
            mol: RDKit molecule
            positions: List of candidate pseudoatom positions
            donor_acceptor_idx: Index of the donor/acceptor atom to exclude from checking
        """
        filtered_positions = []
        
        # Get positions of all non-hydrogen atoms except the donor/acceptor
        atom_positions = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            # Skip hydrogens and the donor/acceptor atom
            if atom.GetAtomicNum() != 1 and i != donor_acceptor_idx:
                atom_positions.append(list(mol.GetConformer().GetAtomPosition(i)))
        
        atom_positions = np.array(atom_positions)

        min_distance = 1.5 if is_donor else 1.8  # Threshold for minimum distance

        for pos in positions:
            distances = np.linalg.norm(atom_positions - pos, axis=1)
            if np.all(distances > min_distance):
                filtered_positions.append(pos)

        return filtered_positions



