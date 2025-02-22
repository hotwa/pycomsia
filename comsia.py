from src.DataLoader import DataLoader
from src.MoleculeAligner import MoleculeAligner
from src.MolecularGridCalculator import MolecularGridCalculator
from src.MolecularFieldCalculator import MolecularFieldCalculator
from src.MolecularVisualizer import MolecularVisualizer
from src.PLSAnalysis import PLSAnalysis
from src.ContourPlotVisualizer import ContourPlotVisualizer
import os
import argparse
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


def main(input_file, sdf_activity, predict_file, grid_resolution, grid_padding, num_components, column_filter, visualization, fields):
    # Create the directory system for results
    current_time = datetime.now()
    directory_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = os.path.join(os.getcwd(), directory_name)
    os.makedirs(output_directory, exist_ok=True)
    subdirectories = ["Contour_Plots", "PLS_Analysis", "Alignments", "Field_Plots"]
    for subdir in subdirectories:
        os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)
    # Load data
    # Check the file extension of input_file
    if input_file.endswith('.csv'):
        # Load data for CSV file
        if predict_file is not None:
            predict_smiles_list, _ = data_loader.load_data(predict_file, is_training=False)
            train_smiles_list, train_activities = data_loader.load_data(input_file, is_training=True)
        else:
            train_smiles_list, train_activities = data_loader.load_data(input_file, is_training=True)

    elif input_file.endswith('.sdf'):
        # Load data for SDF file
        if predict_file is not None:
            predict_smiles_list, predict_mols, _ = data_loader.load_sdf_data(predict_file, is_training=False)
            train_smiles_list, train_mols, train_activities = data_loader.load_sdf_data(input_file, sdf_activity, is_training=True)
            aligned_mols = train_mols + predict_mols
        else:
            train_smiles_list, train_mols, train_activities = data_loader.load_sdf_data(input_file, sdf_activity, is_training=True)
            aligned_mols = train_mols
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or SDF file.")


    # Align molecules
    if input_file.endswith('.csv'):
        if predict_file is not None:
            aligned_mols = aligner.align_molecules(train_smiles_list, predict_smiles_list)
        else:
            aligned_mols = aligner.align_molecules(train_smiles_list)
    else:
        pass
    train_aligned_mols = [mol for mol, is_training in aligned_mols if is_training]
    if visualization == True:
        visualizer.visualize_aligned_molecules(
            train_aligned_mols, output_directory)
    else:
        pass
    # Calculate grid
    grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(aligned_mols, grid_resolution, grid_padding)
    # Calculate fields
    all_field_values = field_calculator.calc_field(aligned_mols, grid_spacing, grid_dimensions, grid_origin)
    # Visualize fields

    visual_field = all_field_values['train_fields']
    
    first_molecule_fields = {
        field: visual_field[f"{field}_field"][0] for field in FIELD_OPTIONS[fields]
    }
    if visualization == True:
        visualizer.visualize_field(train_aligned_mols[0],
            grid_spacing, grid_dimensions, grid_origin,
            first_molecule_fields, output_directory)
    else:
        pass
    # Convert fields for PLS (using only the selected fields)
    train = {k: v for k, v in all_field_values["train_fields"].items() if k.replace("_field", "") in FIELD_OPTIONS[fields]}
    pred = {k: v for k, v in all_field_values["pred_fields"].items() if k.replace("_field", "") in FIELD_OPTIONS[fields]} if predict_file else None

    if predict_file is not None:
        pls_model.convert_fields_to_X(train, pred, filter=column_filter)
    else:
        pls_model.convert_fields_to_X(train, filter=column_filter)
    # Perform PLS analysis
    pls_model.perform_loo_analysis(train_activities, max_components=num_components)
    if predict_file is not None:
        pls_model.fit_final_model(train_activities, test_size=0.2, predict_smiles_list=predict_smiles_list)
    else:
        pls_model.fit_final_model(train_activities, test_size=0.2)
    pls_model.export_metrics_to_csv(output_directory)
    pls_model.export_predictions_and_residuals(output_directory)
    pls_model.plot_results(output_directory)
    coefficients = pls_model.get_coefficient_fields()
    # Visualize coefficients
    if visualization == True:
        significant_ranges = contour_visualizer.calculate_significant_ranges(coefficients)
        contour_visualizer.visualize_contour_plots(train_aligned_mols[0], coefficients, grid_dimensions, grid_origin, grid_spacing, output_directory, significant_ranges)
    else:
        pass
    
def create_parser():
    """
    Creates an ArgumentParser for molecular field analysis.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run molecular field analysis.")

    # Create argument groups
    required = parser.add_argument_group('required arguments')
    input_group = parser.add_argument_group('input options')
    grid_group = parser.add_argument_group('grid options')
    analysis_group = parser.add_argument_group('analysis options')

    # Required arguments
    required.add_argument("--train_file", type=str, 
                         help="CSV file containing SMILES and activity data or an SDF file containing activities. SDF molecules MUST be prealigned.", 
                         required=True)

    # Input options
    input_group.add_argument("--predict_file", type=str, 
                            help="Path to the input CSV file containing SMILES or SDF file for prediction.")
    input_group.add_argument("--sdf_activity", type=str, 
                            help="Activity to use for SDF file. Required if using an SDF file.")

    # Grid options
    grid_group.add_argument("--grid_resolution", type=float, default=1.0, 
                           help="Resolution of the grid used for field calculation. (default: 1.0)")
    grid_group.add_argument("--grid_padding", type=float, default=3.0, 
                           help="Padding of the grid used for field calculation. (default: 3.0)")

    # Analysis options
    analysis_group.add_argument("--fields", type=str, default="all", choices=FIELD_OPTIONS.keys(),
                                help=f"Fields to use for analysis. Options: {', '.join(FIELD_OPTIONS.keys())} (default: all)")
    analysis_group.add_argument("--num_components", type=int, default=12, 
                              help="Number of components for PLS analysis. (default: 12)")
    analysis_group.add_argument("--column_filter", type=float, default=0.0, 
                              help="Column filtering. (default: 0.0)")
    analysis_group.add_argument("--disable_visualization", action='store_true',
                              help="Disable visualization (default: False)")

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    main(args.train_file, args.sdf_activity, args.predict_file,
         args.grid_resolution, args.grid_padding, args.num_components, 
         args.column_filter, not args.disable_visualization, args.fields)
