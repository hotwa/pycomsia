import pickle
from dataclasses import dataclass, field
from typing import Dict, Any
from rdkit import Chem

@dataclass
class SerializedMolecule:
    """
    用于存储 RDKit 分子对象的序列化信息以及相关属性信息。

    Attributes:
        binary: 分子的二进制结构信息（通过 mol.ToBinary() 获取）
        properties: 分子的属性字典（通过 mol.GetPropsAsDict() 获取）
        name: 分子名称，对应分子属性 '_Name'，若不存在则使用默认值 "unknown"
    """
    binary: bytes
    properties: Dict[str, Any] = field(default_factory=dict)
    name: str = "unknown"
    
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> "SerializedMolecule":
        """
        根据一个 RDKit 分子对象构造 SerializedMolecule 对象。
        
        参数:
            mol: 待序列化的 RDKit 分子对象
        返回:
            SerializedMolecule 对象，其 binary 字段保存分子结构，
            properties 字段保存分子的全部属性（通过 GetPropsAsDict 获取），
            name 字段为分子属性 '_Name' 值（如果存在）。
        """
        # 获取二进制结构数据
        binary = mol.ToBinary()
        # 获取所有属性
        properties = mol.GetPropsAsDict()
        # 获取 _Name 属性，如不存在则默认为 "unknown"
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else "unknown"
        return cls(binary=binary, properties=properties, name=name)
    
    def to_mol(self) -> Chem.Mol:
        """
        将存储的二进制数据转换回 RDKit 的 Mol 对象，并恢复属性信息。
        
        返回:
            带有属性恢复的 RDKit 分子对象
        """
        # 通过构造函数重建分子对象
        mol = Chem.Mol(self.binary)
        # 重新设置属性（注意：属性值均保存为字符串）
        for key, value in self.properties.items():
            mol.SetProp(key, str(value))
        # 确保 '_Name' 属性也被设置（优先使用 self.name）
        mol.SetProp('_Name', self.name)
        return mol
    
    def __repr__(self) -> str:
        return f"<SerializedMolecule name={self.name} properties={self.properties}>"


if __name__ == '__main__':
    # 创建一个简单分子对象
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp('_Name', 'Ethanol')
    mol.SetProp('MolecularWeight', '46.07')
    
    # 通过 from_mol 方法构造 SerializedMolecule 对象
    serialized_mol = SerializedMolecule.from_mol(mol)
    print("序列化对象：", serialized_mol)
    
    # 反序列化，恢复为 RDKit Mol 对象
    mol_restored = serialized_mol.to_mol()
    print("恢复后的分子属性：", mol_restored.GetPropsAsDict(), mol_restored.GetProp('_Name'))