# STEP Assembly Tolerance Analysis with 3D Visualization

A Python application that analyzes STEP assembly files to identify geometric features, detect contact surfaces between parts, and determine tolerancing requirements. Includes 3D visualization to highlight contact surfaces for easy verification.

## Features

✅ **Feature Identification**: Automatically identifies geometric features (planes, cylinders, holes)  
✅ **Contact Detection**: Finds surfaces in contact between different parts  
✅ **Tolerance Recommendations**: Provides specific ISO fit and GD&T recommendations  
✅ **3D Visualization**: Interactive 3D rendering with contact surfaces highlighted in red  
✅ **Contact Analysis**: Detailed charts showing contact types, distances, and areas  
✅ **Multiple Output Formats**: Console reports, JSON data, and PNG visualizations  

## Installation

```bash
pip install -r requirements.txt
```

For system-wide installation (Ubuntu/Debian):
```bash
pip install --break-system-packages -r requirements.txt
```

## Quick Start

### 1. Run the Demo
```bash
python demo.py
```
This creates a sample assembly, analyzes it, and generates visualizations.

### 2. Analyze Your STEP File
```bash
# Basic analysis
python analyze_step.py your_assembly.step

# With custom tolerance
python analyze_step.py your_assembly.step 2.0

# With 3D visualization
python analyze_step.py your_assembly.step 2.0 --visualize
```

### 3. Quick Visualization
```bash
python visualize_step.py your_assembly.step 5.0
```

## Understanding the 3D Visualization

The visualization shows:
- **Different colored parts**: Blue, green, yellow represent different components
- **RED surfaces**: Features that are in contact with other parts
- **RED stars (★)**: Contact center points
- **RED lines**: Connect features that are touching

### Example Visualization

See `sample_assembly_3d_view.png` for an example showing:
- 3 parts (base plate, shaft, top plate)
- Contact surfaces highlighted in red where parts touch
- Contact points marked with stars

### Contact Analysis Report

The `sample_assembly_contact_report.png` shows:
1. **Contact Types Distribution**: Pie chart of different contact types
2. **Distance Distribution**: Histogram of distances between contacts
3. **Contact Areas**: Bar chart comparing contact surface areas
4. **Interaction Matrix**: Heatmap showing which parts contact each other

## How It Works

### 1. Feature Identification
The tool identifies:
- **Flat Faces**: Planar surfaces for mating
- **Cylindrical Surfaces**: Shafts, pins, bosses
- **Holes**: Circular openings for fasteners or shafts

### 2. Contact Detection
Contacts are detected when:
- Features are from different parts
- Features are within the tolerance distance
- Bounding boxes overlap

Tolerance parameter (in mm) determines contact threshold:
- `0.01-0.1`: Tight assemblies (precision fits)
- `0.1-1.0`: Normal assemblies
- `1.0-10.0`: Loose assemblies or visualization

### 3. Tolerance Evaluation
The tool identifies required tolerances:

**CRITICAL** - Essential for function:
- Hole-shaft fits (provides ISO fit designations like H7/h6)

**IMPORTANT** - Significant for quality:
- Mating surfaces (flatness, profile)
- Perpendicularity requirements
- Parallelism requirements

**STANDARD** - General tolerancing:
- Edge contacts (positional tolerance)
- Surface contacts (profile tolerance)

## Command Reference

### analyze_step.py
Main analysis tool with full reporting and optional visualization.

```bash
python analyze_step.py <step_file> [tolerance] [--visualize] [--no-show]

Arguments:
  step_file     Path to STEP assembly file (.step or .stp)
  tolerance     Contact detection distance in mm (default: 0.01)
  --visualize   Generate 3D visualizations
  --no-show     Save images without displaying (for batch processing)

Examples:
  python analyze_step.py assembly.step
  python analyze_step.py assembly.step 2.0 --visualize
  python analyze_step.py assembly.step 1.0 --visualize --no-show
```

Outputs:
- Console report with tolerance recommendations
- `<filename>_analysis.json`: Complete analysis data
- `<filename>_3d_view.png`: 3D visualization (if --visualize)
- `<filename>_contact_report.png`: Analysis charts (if contacts found)

### visualize_step.py
Quick visualization tool for rapid visual inspection.

```bash
python visualize_step.py <step_file> [tolerance]

Arguments:
  step_file     Path to STEP assembly file
  tolerance     Contact detection distance in mm (default: 1.0)

Example:
  python visualize_step.py assembly.step 5.0
```

Outputs:
- `<filename>_visualization.png`: 3D view
- `<filename>_contact_analysis.png`: Contact charts

### create_sample.py
Generates a sample assembly for testing.

```bash
python create_sample.py
```

Creates:
- `sample_assembly.step`: Simple 3-part assembly
- `sample_base.step`: Individual base part
- `sample_shaft.step`: Individual shaft part

## Tolerance Recommendations

The tool provides specific recommendations based on contact type:

| Contact Type | Tolerance Type | Recommendation |
|--------------|----------------|----------------|
| Hole-Shaft Fit | ISO Fit | H7/h6, H8/f7, etc. (size-dependent) |
| Mating Surfaces | Flatness/Profile | ±0.05-0.1 mm |
| Perpendicular Contact | Perpendicularity | ±0.1 mm |
| Line Contact | Parallelism | ±0.1 mm |
| Edge Contact | Position | ±0.2 mm |
| General Contact | Profile | ±0.15 mm |

## File Descriptions

### Core Application
- `analyze_step.py` - Main analysis script with visualization
- `feature_analyzer.py` - Identifies geometric features from STEP files
- `contact_detector.py` - Detects contacts between parts
- `tolerance_evaluator.py` - Determines tolerance requirements
- `visualizer.py` - 3D visualization engine

### Utilities
- `visualize_step.py` - Standalone quick visualization tool
- `create_sample.py` - Sample assembly generator
- `demo.py` - Complete workflow demonstration
- `requirements.txt` - Python dependencies

### Example Outputs
- `sample_assembly_3d_view.png` - Example visualization
- `sample_assembly_contact_report.png` - Example analysis charts

## Troubleshooting

### No contacts detected
- Increase the tolerance parameter (try 1.0, 5.0, or 10.0 mm)
- Parts may not actually be touching in the assembly
- Check that the STEP file contains multiple parts

### Visualization not displaying
- Make sure matplotlib is installed: `pip install matplotlib`
- Use `--no-show` flag to save images without display
- Check that X11/display server is available for interactive display

### STEP file loading errors
- Ensure file is valid STEP format (.step or .stp)
- Try re-exporting the STEP file from your CAD software
- Verify file is not corrupted

## Technical Details

**Supported STEP Versions**: AP203, AP214, AP242  
**Feature Detection**: Uses OpenCASCADE geometry kernel via CadQuery  
**Visualization**: Matplotlib with 3D surface triangulation  
**Contact Algorithm**: Bounding box overlap + distance threshold  

## License

This tool is provided as-is for engineering analysis purposes.

## Support

For issues or questions, refer to USAGE.txt for detailed instructions.
