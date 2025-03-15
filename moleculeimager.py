import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import sys
import os

def process_sdf(sdf_file):
    # Read molecules from SDF
    suppl = Chem.SDMolSupplier(sdf_file)
    molecules = []
    names = []
    
    for idx, mol in enumerate(suppl):
        try:
            if mol is not None:
                mol = Chem.RemoveHs(mol)
                # Generate 2D coordinates if not present
                AllChem.Compute2DCoords(mol)
                
                # Try to get compound name from PubChem
                try:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    compounds = pcp.get_compounds(smiles, 'smiles')
                    if compounds:
                        name = compounds[0].iupac_name[:50]  # Truncate long names
                    else:
                        name = f"Compound_{idx}"
                except:
                    name = f"Compound_{idx}"
                    
                molecules.append(mol)
                names.append(name)
            else:
                print(f"Warning: Molecule at position {idx} could not be loaded")
        except Exception as e:
            print(f"Error processing molecule at position {idx}: {str(e)}")
    
    if not molecules:
        raise ValueError("No valid molecules found in the SDF file")
    
    return molecules, names

def create_image(molecules, names, output_file, mols_per_row=5):
    try:
        # Create image grid with explicit drawing options
        img = Draw.MolsToGridImage(
            molecules,
            legends=names,
            molsPerRow=mols_per_row,
            subImgSize=(300, 300)
            )
        
        # Save image
        img.save(output_file)
        print(f"Successfully saved image to {output_file}")
        
    except Exception as e:
        print(f"Error creating image: {str(e)}")
        raise

def main():
    # Check if an SDF file was provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python moleculeimager.py <sdf_file>")
        sys.exit(1)

    # Get the input SDF file path
    sdf_file = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(sdf_file):
        print(f"Error: File '{sdf_file}' not found")
        sys.exit(1)

    # Generate output filename by replacing .sdf extension with .png
    output_file = os.path.splitext(sdf_file)[0] + '.png'

    try:
        print(f"Processing SDF file: {sdf_file}")
        molecules, names = process_sdf(sdf_file)
        print(f"Found {len(molecules)} valid molecules")
        
        print("Creating image...")
        create_image(molecules, names, output_file)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
