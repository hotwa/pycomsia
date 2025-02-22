import numpy as np
import pyvista as pv
from rdkit import Chem

class ContourPlotVisualizer:
    def __init__(self):
        self.significant_ranges = None
        
    def calculate_significant_ranges(self, reconstructed_coeffs, top_percent=0.5, bottom_percent=0.5):
        """Calculate significant value ranges for each field"""
        significant_ranges = {}
        
        for field_name, field_values in reconstructed_coeffs.items():
            flat_values = field_values.flatten()
            sorted_values = np.sort(flat_values)
            
            n = len(sorted_values)
            bottom_index = int(n * bottom_percent / 100)
            top_index = int(n * (1 - top_percent / 100))
            
            low_range = sorted_values[:bottom_index]
            high_range = sorted_values[top_index:]
            
            significant_ranges[field_name] = {
                'low': (np.min(low_range), np.max(low_range)),
                'high': (np.min(high_range), np.max(high_range))
            }
        
        self.significant_ranges = significant_ranges
        return significant_ranges
    
    def visualize_contour_plots(self, mol, reconstructed_coeffs, grid_dimensions, 
                                grid_origin, grid_spacing, output, significant_ranges=None):
        """Visualize molecular fields using marching cubes algorithm with field-specific colors and legends"""
        
        color_map = {
            'steric_field': {
                'low': 'green',
                'high': 'yellow'
            },
            'electrostatic_field': {
                'low': 'blue', 
                'high': 'red'
            },
            'hydrophobic_field': {
                'low': 'purple', 
                'high': 'orange'
            },
            'hbond_acceptor_field': {
                'low': 'purple', 
                'high': 'teal'
            },
            'hbond_donor_field': {
                'low': 'grey', 
                'high': 'yellow'
            }
        }
        
        if significant_ranges is None:
            significant_ranges = self.significant_ranges
            
        for field_name, field_values in reconstructed_coeffs.items():
            if field_values.ndim == 1:
                field_values = field_values.reshape(grid_dimensions)
            
            grid = pv.ImageData()
            grid.dimensions = np.array(grid_dimensions)
            grid.origin = grid_origin
            grid.spacing = grid_spacing
            grid.point_data[field_name] = field_values.flatten(order="F")
            
            plotter = pv.Plotter(off_screen=True)
            pv.global_theme.allow_empty_mesh = True
            low_range = significant_ranges[field_name]['low']
            high_range = significant_ranges[field_name]['high']
            
            # Get colors for the specific field
            low_color = color_map.get(field_name, {}).get('low', 'blue')
            high_color = color_map.get(field_name, {}).get('high', 'red')
            
            # Create isosurfaces for low values
            contours_low = grid.contour(
                [low_range[0], low_range[1]],
                scalars=field_name,
                method='marching_cubes'
            )
            contours_low = contours_low.interpolate(grid)
            plotter.add_mesh(contours_low, color=low_color, opacity=0.7, smooth_shading=True)
            
            # Create isosurfaces for high values
            contours_high = grid.contour(
                [high_range[0], high_range[1]],
                scalars=field_name,
                method='marching_cubes'
            )
            contours_high = contours_high.interpolate(grid)
           
            plotter.add_mesh(contours_high, color=high_color, opacity=0.7, smooth_shading=True)
            
            # Add molecule visualization
            self._add_molecule_to_plot(plotter, mol)
            
            # Set visualization properties
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.2)
            plotter.background_color = 'white'
            
            
            filename = f"{output}/Contour_Plots/{field_name}_contourplot.png"
            plotter.screenshot(filename, scale=5)
            print(f"Saved contour plot for {field_name} field to {filename}")
            plotter.close()
    
    def _add_molecule_to_plot(self, plotter, mol):
        """Add molecule representation to the plot"""
        mol = Chem.RemoveHs(mol)
        conformer = mol.GetConformer()
       
        
        # Add atoms
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            coord = np.array(conformer.GetAtomPosition(idx))
            atomic_num = atom.GetAtomicNum()
            
            sphere = pv.Sphere(radius=0.2, phi_resolution=20, theta_resolution=20)
            atoms = pv.PolyData(coord).glyph(geom=sphere)
            color = self._get_atom_color(atomic_num)
            plotter.add_mesh(atoms, color=color)
        
        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start = conformer.GetAtomPosition(begin_idx)
            end = conformer.GetAtomPosition(end_idx)
            
            color = self._get_bond_color(bond)
            line = pv.Tube(pointa=start, pointb=end, radius=0.1, n_sides=15)
            plotter.add_mesh(line, color=color, smooth_shading=True)
    
    def _get_atom_color(self, atomic_num):
        """Get color for specific atom type"""
        colors = {
            1: 'white',     # Hydrogen
            6: 'silver',    # Carbon
            7: 'lightblue', # Nitrogen
            8: 'red',       # Oxygen
        }
        return colors.get(atomic_num, 'lightgray')
    
    def _get_bond_color(self, bond):
        """Get color for specific bond type"""
        bond_types = {
            1.0: 'silver',     # Single bond
            1.5: 'silver',  # Aromatic
            2.0: 'silver',  # Double bond
            3.0: 'silver'         # Triple bond
        }
        return bond_types.get(bond.GetBondTypeAsDouble(), 'silver')