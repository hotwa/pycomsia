from src.DataLoader import DataLoader
from src.MoleculeAligner import MoleculeAligner
from src.MolecularGridCalculator import MolecularGridCalculator
from src.MolecularFieldCalculator import MolecularFieldCalculator
from src.MolecularVisualizer import MolecularVisualizer
from src.PLSAnalysisTestSets import PLSAnalysis
from src.ContourPlotVisualizer import ContourPlotVisualizer
import os
from datetime import datetime 
data_loader = DataLoader()
aligner = MoleculeAligner()
field_calculator = MolecularFieldCalculator()
visualizer = MolecularVisualizer()
grid_calculator = MolecularGridCalculator()
pls_model = PLSAnalysis()
contour_visualizer = ContourPlotVisualizer()

FIELD_OPTIONS = {
    "all": ["steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor"],
    "SE": ["steric", "electrostatic"],
    "SED": ["steric", "electrostatic", "hbond_donor"],
    "SEA": ["steric", "electrostatic", "hbond_acceptor"],
    "SEAD": ["steric", "electrostatic", "hbond_acceptor", "hbond_donor"],
    "SEDA": ["steric", "electrostatic", "hbond_acceptor", "hbond_donor"],
    "SEH": ["steric", "electrostatic", "hydrophobic"]
}


def main(train_file, test_file, grid_resolution, grid_padding, num_components, column_filter, fields):
    # Extract the base name of the training file (e.g., "ACE" from "ACE_train.sdf")
    train_file_base = os.path.basename(train_file).split('_')[0]
    current_time = datetime.now()
    directory_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # Create the directory system for results using the base name
    output_directory = os.path.join(os.getcwd(), train_file_base+"_"+fields+"_"+ directory_name)
    os.makedirs(output_directory, exist_ok=True)
    subdirectories = ["Contour_Plots", "PLS_Analysis", "Alignments", "Field_Plots"]
    for subdir in subdirectories:
        os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)
    # Load data
    # Check the file extension of input_file

    train_smiles_list, train_mols, train_activities = data_loader.load_sdf_data(train_file, activity_property = "Activity", is_training=True)
    test_smiles_list, test_mols, test_activities = data_loader.load_sdf_data(test_file, activity_property = "Activity", is_training=True)
    aligned_mols = train_mols + test_mols

    train_aligned_mols = [mol for mol, is_training in aligned_mols if is_training]
    visualizer.visualize_aligned_molecules(
            train_aligned_mols, output_directory)

    # Calculate grid
    grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(aligned_mols, grid_resolution, grid_padding)
    # Calculate fields
    train_fields = field_calculator.calc_field(train_mols, grid_spacing, grid_dimensions, grid_origin)
    test_fields = field_calculator.calc_field(test_mols, grid_spacing, grid_dimensions, grid_origin)
    # Visualize fields
    visual_field = train_fields['train_fields']
    first_molecule_fields = {
        field: visual_field[f"{field}_field"][0] for field in FIELD_OPTIONS[fields]
    }
 
    visualizer.visualize_field(train_aligned_mols[0],
        grid_spacing, grid_dimensions, grid_origin,
        first_molecule_fields, output_directory)
    # Convert fields for PLS
    train = {k: v for k, v in train_fields["train_fields"].items() if k.replace("_field", "") in FIELD_OPTIONS[fields]}
    test = {k: v for k, v in test_fields["train_fields"].items() if k.replace("_field", "") in FIELD_OPTIONS[fields]} if test_file else None
    pls_model.convert_fields_to_X(train, test, filter=column_filter)
    # Perform PLS analysis
    pls_model.perform_loo_analysis(train_activities, max_components=num_components)
    pls_model.fit_final_model(train_activities, test_activities)
    pls_model.export_metrics_to_csv(output_directory)
    pls_model.export_predictions_and_residuals(output_directory)
    pls_model.plot_results(output_directory)
    coefficients = pls_model.get_coefficient_fields()
    # Visualize coefficients
    significant_ranges = contour_visualizer.calculate_significant_ranges(coefficients)
    contour_visualizer.visualize_contour_plots(train_aligned_mols[0], coefficients, grid_dimensions, grid_origin, grid_spacing, output_directory, significant_ranges)

    


import os

# Directory containing the SDF files
sdfs_dir = "sdfs"

# Parameters for the main function
grid_resolution = 1
grid_padding = 4
num_components = 12
column_filter = 0.0


# List all files in the directory
files = os.listdir(sdfs_dir)

# Filter out only the train and test files
train_files = [f for f in files if f.endswith("_train.sdf")]
test_files = [f for f in files if f.endswith("_test.sdf")]

# Match train and test files based on their prefixes
for train_file in train_files:
    prefix = train_file.replace("_train.sdf", "")
    corresponding_test_file = f"{prefix}_test.sdf"
    if train_file == "ACE_train.sdf":
        fields = "SEH"
    elif train_file == "AChE_train.sdf":
        fields = "all"
    elif train_file == "THERM_train.sdf":
        fields = "SEAD"
    elif train_file == "THR_train.sdf":
        fields = "all"
    elif train_file == "STEROIDS_train.sdf":
        fields = "all"
    elif train_file == "DAT_train.sdf":
        fields = "all"
    elif train_file == "CCR5_train.sdf":
        fields = "all"
    elif train_file == "ATA_train.sdf":
        fields = "all"
    if corresponding_test_file in test_files:
        # Construct full paths
        train_path = os.path.join(sdfs_dir, train_file)
        test_path = os.path.join(sdfs_dir, corresponding_test_file)
        
        # Execute the main function
        print(f"Processing: {train_path} and {test_path}")
        main(train_path, test_path, grid_resolution, grid_padding, num_components, column_filter, fields)
    else:
        print(f"No matching test file found for {train_file}")
