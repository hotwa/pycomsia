# Py-CoMSIA: Pythonic CoMSIA 3D QSAR

`comsia.py` is a Python implementation of Comparative Molecular Similarity Indices Analysis (CoMSIA), a 3D Quantitative Structure-Activity Relationship (QSAR) method. This tool allows you to analyze molecular fields and predict biological activities based on molecular structures.

## Features

* **CoMSIA Field Calculation:** Calculates steric, electrostatic, hydrophobic, hydrogen bond donor, and hydrogen bond acceptor fields.
* **PLS Regression:** Utilizes Partial Least Squares (PLS) regression for building QSAR models.
* **Flexible Input:** Supports both CSV files (SMILES and activity data) and pre-aligned SDF files.
* **Grid-Based Analysis:** Configurable grid resolution and padding for field calculations.
* **Field Selection:** Allows you to select specific fields for analysis.
* **Visualization:** (Optional) Visualization of the CoMSIA fields and PLS results.
* **Prediction:** Predict activities for new compounds based on the trained model.
* **Column filtering:** option to filter out columns with low variance.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/clhaga/pycomsia
    cd pycomsia
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt 
    ```

## Usage

```bash
python comsia.py --train_file <train_file> [options]
```

## Arguments
--train_file (required): Path to the training data. Can be a CSV file with SMILES and activity data or an SDF file containing pre-aligned molecules and activity data.

--predict_file: Path to the input CSV or SDF file for prediction.

--sdf_activity: Activity to use for SDF file. Required if using an SDF file.

--grid_resolution: Resolution of the grid used for field calculation. (default: 1.0)

--grid_padding: Padding of the grid used for field calculation. (default: 3.0)

--fields: Fields to use for analysis. Options: steric, electrostatic, hydrophobic, donor, acceptor, all. (default: all)

--num_components: Number of components for PLS analysis. (default: 12)

--column_filter: Column filtering. (default: 0.0)

--disable_visualization: Disable visualization. (default: False)

## Data Format
CSV:
One molecule per row.
A column for SMILES strings.
A column for the activity data.

SDF:
Molecules should be pre-aligned.
The SDF file must contain a property field corresponding to the activity data.
Use --sdf_activity to specify the property name.

## Tests from publication

To run the examples from the publication, simply execute the following:

```bash
python comsiatest.py
```
