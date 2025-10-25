#!/usr/bin/env python3
"""
STEP Assembly Tolerance Analysis Tool

Analyzes STEP assembly files to:
1. Identify geometric features
2. Detect contact between features
3. Determine tolerancing requirements
4. Visualize assemblies with contact highlighting

Usage:
    python analyze_step.py <step_file_path> [tolerance] [--visualize] [--no-show]
"""

import sys
import os
from typing import Optional
from feature_analyzer import FeatureAnalyzer
from contact_detector import ContactDetector
from tolerance_evaluator import ToleranceEvaluator
import json


class STEPAnalyzer:
    """Main analyzer class that coordinates the analysis pipeline"""
    
    def __init__(self, step_file_path: str, tolerance: float = 0.01):
        """
        Initialize the analyzer
        
        Args:
            step_file_path: Path to the STEP file
            tolerance: Distance tolerance for contact detection (mm)
        """
        self.step_file_path = step_file_path
        self.tolerance = tolerance
        self.feature_analyzer = None
        self.contact_detector = None
        self.tolerance_evaluator = None
    
    def analyze(self) -> dict:
        """
        Perform complete analysis of the STEP file
        
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "file": self.step_file_path,
            "success": False,
            "error": None,
            "features": [],
            "contacts": [],
            "tolerance_requirements": []
        }
        
        try:
            # Step 1: Load and analyze features
            print(f"Loading STEP file: {self.step_file_path}")
            self.feature_analyzer = FeatureAnalyzer(self.step_file_path)
            
            print("Identifying features...")
            features = self.feature_analyzer.identify_features()
            
            print(f"Found {len(features)} features in {self.feature_analyzer.get_part_count()} parts")
            
            # Step 2: Detect contacts
            print("Detecting contacts between features...")
            self.contact_detector = ContactDetector(features, tolerance=self.tolerance)
            contacts = self.contact_detector.detect_contacts()
            
            print(f"Detected {len(contacts)} contacts")
            
            # Step 3: Evaluate tolerance requirements
            print("Evaluating tolerance requirements...")
            self.tolerance_evaluator = ToleranceEvaluator(contacts)
            requirements = self.tolerance_evaluator.evaluate_tolerances()
            
            print(f"Identified {len(requirements)} tolerance requirements")
            
            # Compile results
            results["success"] = True
            results["features"] = self._serialize_features(features)
            results["contacts"] = self._serialize_contacts(contacts)
            results["tolerance_requirements"] = self._serialize_requirements(requirements)
            results["statistics"] = self._get_statistics(features, contacts, requirements)
            
        except Exception as e:
            results["error"] = str(e)
            print(f"Error during analysis: {e}")
        
        return results
    
    def _serialize_features(self, features):
        """Convert features to serializable format"""
        return [
            {
                "id": f.id,
                "part_id": f.part_id,
                "type": f.feature_type.value,
                "position": f.position.tolist(),
                "normal": f.normal.tolist() if f.normal is not None else None,
                "dimensions": f.dimensions,
                "bounding_box": f.bounding_box
            }
            for f in features
        ]
    
    def _serialize_contacts(self, contacts):
        """Convert contacts to serializable format"""
        return [
            {
                "feature1_id": c.feature1.id,
                "feature2_id": c.feature2.id,
                "feature1_type": c.feature1.feature_type.value,
                "feature2_type": c.feature2.feature_type.value,
                "contact_type": c.contact_type,
                "distance": float(c.distance),
                "contact_area": float(c.contact_area)
            }
            for c in contacts
        ]
    
    def _serialize_requirements(self, requirements):
        """Convert tolerance requirements to serializable format"""
        return [
            {
                "contact": {
                    "feature1_id": r.contact.feature1.id,
                    "feature2_id": r.contact.feature2.id,
                    "contact_type": r.contact.contact_type
                },
                "tolerance_type": r.tolerance_type,
                "reason": r.reason,
                "recommended_tolerance": r.recommended_tolerance,
                "priority": r.priority
            }
            for r in requirements
        ]
    
    def _get_statistics(self, features, contacts, requirements):
        """Generate summary statistics"""
        stats = self.tolerance_evaluator.get_summary_statistics()
        stats.update({
            "total_features": len(features),
            "total_parts": self.feature_analyzer.get_part_count(),
            "total_contacts": len(contacts)
        })
        
        # Get counts by type
        contacts_by_type = self.contact_detector.get_contacts_by_type()
        stats["contacts_by_type"] = {k: len(v) for k, v in contacts_by_type.items()}
        
        return stats
    
    def print_report(self, results: dict):
        """Print a formatted report of the analysis"""
        
        print("\n" + "="*80)
        print("STEP ASSEMBLY TOLERANCE ANALYSIS REPORT")
        print("="*80)
        
        if not results["success"]:
            print(f"\nERROR: {results['error']}")
            return
        
        # Summary statistics
        stats = results["statistics"]
        print(f"\nFILE: {results['file']}")
        print(f"\nSUMMARY:")
        print(f"  - Parts: {stats['total_parts']}")
        print(f"  - Features: {stats['total_features']}")
        print(f"  - Contacts: {stats['total_contacts']}")
        print(f"  - Tolerance Requirements: {stats['total_requirements']}")
        
        # Contact types
        if stats.get("contacts_by_type"):
            print(f"\nCONTACT TYPES:")
            for contact_type, count in stats["contacts_by_type"].items():
                print(f"  - {contact_type}: {count}")
        
        # Tolerance requirements by priority
        print(f"\nTOLERANCE REQUIREMENTS BY PRIORITY:")
        print(f"  - Critical: {stats['critical']}")
        print(f"  - Important: {stats['important']}")
        print(f"  - Standard: {stats['standard']}")
        
        # Detailed tolerance requirements
        if results["tolerance_requirements"]:
            print(f"\nDETAILED TOLERANCE REQUIREMENTS:")
            print("-" * 80)
            
            for idx, req in enumerate(results["tolerance_requirements"], 1):
                print(f"\n{idx}. {req['tolerance_type']} [{req['priority']}]")
                print(f"   Contact: Feature {req['contact']['feature1_id']} <-> "
                      f"Feature {req['contact']['feature2_id']}")
                print(f"   Type: {req['contact']['contact_type']}")
                print(f"   Reason: {req['reason']}")
                print(f"   Recommendation: {req['recommended_tolerance']}")
        
        print("\n" + "="*80)
        
        # Notifications for critical requirements
        critical_count = stats['critical']
        if critical_count > 0:
            print(f"\n[!] WARNING: {critical_count} CRITICAL tolerance requirement(s) detected!")
            print("    These require immediate attention for proper assembly function.")
        
        if stats['important'] > 0:
            print(f"\n[*] NOTICE: {stats['important']} IMPORTANT tolerance requirement(s) detected.")
            print("    These should be addressed to ensure quality assembly.")
        
        print("\n")


def main():
    """Main entry point"""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_step.py <step_file_path> [tolerance] [--visualize] [--no-show]")
        print("\nArguments:")
        print("  step_file_path  : Path to the STEP assembly file (.step or .stp)")
        print("  tolerance       : (Optional) Distance tolerance in mm (default: 0.01)")
        print("  --visualize     : (Optional) Generate 3D visualization of the assembly")
        print("  --no-show       : (Optional) Save visualizations without displaying them")
        print("\nExample:")
        print("  python analyze_step.py assembly.step")
        print("  python analyze_step.py assembly.step 0.05")
        print("  python analyze_step.py assembly.step 2.0 --visualize")
        print("  python analyze_step.py assembly.step 1.0 --visualize --no-show")
        sys.exit(1)
    
    step_file = sys.argv[1]
    
    # Parse remaining arguments
    tolerance = 0.01
    visualize = False
    show_viz = True
    
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--visualize':
            visualize = True
        elif arg == '--no-show':
            show_viz = False
        else:
            try:
                tolerance = float(arg)
            except ValueError:
                print(f"Warning: Invalid argument '{arg}' ignored")
    
    # Check if file exists
    if not os.path.exists(step_file):
        print(f"Error: File '{step_file}' not found!")
        sys.exit(1)
    
    # Run analysis
    analyzer = STEPAnalyzer(step_file, tolerance=tolerance)
    results = analyzer.analyze()
    
    # Print report
    analyzer.print_report(results)
    
    # Save results to JSON
    output_file = os.path.splitext(step_file)[0] + "_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    
    # Generate visualization if requested
    if visualize:
        try:
            from visualizer import AssemblyVisualizer
            
            print("\n" + "="*80)
            print("GENERATING 3D VISUALIZATION")
            print("="*80 + "\n")
            
            # Create visualizer
            viz = AssemblyVisualizer(analyzer.feature_analyzer, analyzer.contact_detector.contacts)
            
            # Generate 3D assembly view
            viz_3d_file = os.path.splitext(step_file)[0] + "_3d_view.png"
            viz.visualize(output_file=viz_3d_file if not show_viz else None, show=show_viz)
            
            if not show_viz:
                print(f"3D visualization saved to: {viz_3d_file}")
            
            # Generate contact report if there are contacts
            if analyzer.contact_detector.contacts:
                report_file = os.path.splitext(step_file)[0] + "_contact_report.png"
                viz.create_contact_report_visualization(
                    output_file=report_file if not show_viz else None
                )
                
                if not show_viz:
                    print(f"Contact report saved to: {report_file}")
            
        except ImportError as e:
            print(f"\nWarning: Could not import visualization module: {e}")
            print("Visualization requires matplotlib. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\nWarning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
