#!/usr/bin/env python3
"""
Standalone STEP Assembly Visualizer

Quickly visualize STEP files with contact surface highlighting

Usage:
    python visualize_step.py <step_file_path> [tolerance]
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_step.py <step_file_path> [tolerance]")
        print("\nArguments:")
        print("  step_file_path  : Path to the STEP assembly file (.step or .stp)")
        print("  tolerance       : (Optional) Distance tolerance in mm (default: 1.0)")
        print("\nExample:")
        print("  python visualize_step.py assembly.step")
        print("  python visualize_step.py assembly.step 2.0")
        sys.exit(1)
    
    step_file = sys.argv[1]
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    if not os.path.exists(step_file):
        print(f"Error: File '{step_file}' not found!")
        sys.exit(1)
    
    try:
        from visualizer import visualize_assembly
        
        # Generate visualization
        base_name = os.path.splitext(step_file)[0]
        output_3d = base_name + "_visualization.png"
        output_report = base_name + "_contact_analysis.png"
        
        print("="*80)
        print(f"VISUALIZING: {step_file}")
        print(f"Tolerance: {tolerance} mm")
        print("="*80 + "\n")
        
        visualize_assembly(
            step_file,
            tolerance=tolerance,
            output_3d=output_3d,
            output_report=output_report,
            show=True
        )
        
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"3D View saved to: {output_3d}")
        print(f"Contact Report saved to: {output_report}")
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
