
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


class DataLoader:
    def __init__(self):
        self.smiles_list = None
        self.activities = None
        self.molecules = None
        
    def load_data(self, csv_file, is_training=True):
        """
        Load SMILES and activity data from CSV
        
        Parameters:
        -----------
        csv_file : str
            Path to the CSV file
        is_training : bool, default=True
            If True, loads both SMILES and activities
            If False, loads only SMILES (for prediction dataset)
            
        Returns:
        --------
        tuple
            If is_training=True: (smiles_list, activities)
            If is_training=False: (smiles_list, None)
        """
        df = pd.read_csv(csv_file)
        self.smiles_list = df['SMILES'].tolist()
        
        if is_training:
            if 'Activity' not in df.columns:
                raise ValueError("Activity column not found in training dataset")
            self.activities = df['Activity'].values
        else:
            self.activities = None
        
        return self.smiles_list, self.activities
    
    def load_sdf_data(self, sdf_file, activity_property=None, is_training=True):
        """
        Load SMILES, 3D molecules, and activity data from SDF

        Parameters:
        -----------
        sdf_file : str
            Path to the SDF file
        activity_property : str, optional
            Name of the property in the SDF file that contains the activity data
            Required if is_training=True
        is_training : bool, default=True
            If True, loads both SMILES and activities
            If False, loads only SMILES (for prediction dataset)

        Returns:
        --------
        tuple
            If is_training=True: (smiles_list, [(mol, is_training)], activities)
            If is_training=False: (smiles_list, [(mol, is_training)], None)
        """
        # Read the SDF file
        print("Reading SDF file...")
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        self.smiles_list = []
        mols_with_flag = []  # List of tuples: (mol, is_training)
        self.activities = []

        for mol in suppl:
            if mol is not None:
                # Generate SMILES from the molecule
                smiles = Chem.MolToSmiles(mol)
                self.smiles_list.append(smiles)

                # Store the 3D molecule with the is_training flag
                mol = Chem.AddHs(mol)
                mols_with_flag.append((mol, is_training))

                # Extract activity if in training mode
                if is_training:
                    if activity_property is None:
                        raise ValueError("Activity property must be specified for training data")
                    if activity_property not in mol.GetPropNames():
                        raise ValueError(f"Activity property '{activity_property}' not found in SDF file")
                    activity = float(mol.GetProp(activity_property))
                    self.activities.append(activity)

        if not is_training:
            self.activities = None

        return self.smiles_list, mols_with_flag, self.activities