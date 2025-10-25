#!/usr/bin/env python3
"""
Diagnostic script to debug contact detection issues
Provides detailed information about features and why contacts may not be detected
"""

import sys
import numpy as np
from feature_analyzer import FeatureAnalyzer, FeatureType
from contact_detector import ContactDetector


def diagnose_assembly(step_file: str, tolerance: float = 2.0):
    """Diagnose why contacts aren't being detected"""
    
    print("="*80)
    print(f"DIAGNOSING: {step_file}")
    print(f"Tolerance: {tolerance} mm")
    print("="*80)
    
    # Load and analyze
    analyzer = FeatureAnalyzer(step_file)
    features = analyzer.identify_features()
    
    print(f"\nParts loaded: {analyzer.get_part_count()}")
    print(f"Total features: {len(features)}")
    
    # Group features by part and type
    print("\n" + "-"*80)
    print("FEATURES BY PART:")
    print("-"*80)
    
    for part_id in range(analyzer.get_part_count()):
        part_features = [f for f in features if f.part_id == part_id]
        print(f"\nPart {part_id}: {len(part_features)} features")
        
        for f in part_features:
            print(f"  Feature {f.id}: {f.feature_type.value}")
            print(f"    Position: ({f.position[0]:.2f}, {f.position[1]:.2f}, {f.position[2]:.2f})")
            if f.normal is not None:
                print(f"    Normal: ({f.normal[0]:.3f}, {f.normal[1]:.3f}, {f.normal[2]:.3f})")
            if f.dimensions:
                for key, val in f.dimensions.items():
                    print(f"    {key}: {val:.3f}")
    
    # Find cylindrical features
    print("\n" + "-"*80)
    print("CYLINDRICAL FEATURES:")
    print("-"*80)
    
    cylinders = [f for f in features if f.feature_type in [FeatureType.CYLINDER, FeatureType.HOLE]]
    
    if len(cylinders) == 0:
        print("⚠️  NO CYLINDRICAL FEATURES FOUND!")
        print("   The assembly may not have any cylinders or holes detected.")
        return
    
    print(f"\nFound {len(cylinders)} cylindrical features")
    
    for cyl in cylinders:
        print(f"\nFeature {cyl.id} (Part {cyl.part_id}): {cyl.feature_type.value}")
        print(f"  Position: ({cyl.position[0]:.2f}, {cyl.position[1]:.2f}, {cyl.position[2]:.2f})")
        print(f"  Radius: {cyl.dimensions.get('radius', 0):.3f} mm")
        if cyl.normal is not None:
            print(f"  Axis: ({cyl.normal[0]:.3f}, {cyl.normal[1]:.3f}, {cyl.normal[2]:.3f})")
    
    # Check pairs of cylinders from different parts
    if len(cylinders) >= 2:
        print("\n" + "-"*80)
        print("CYLINDER PAIR ANALYSIS:")
        print("-"*80)
        
        for i, c1 in enumerate(cylinders):
            for c2 in cylinders[i+1:]:
                if c1.part_id == c2.part_id:
                    continue
                
                print(f"\n📊 Comparing Feature {c1.id} and Feature {c2.id}:")
                
                # Check radii
                r1 = c1.dimensions.get('radius', 0)
                r2 = c2.dimensions.get('radius', 0)
                clearance = abs(r1 - r2)
                
                print(f"   Radii: {r1:.3f} mm vs {r2:.3f} mm")
                print(f"   Clearance: {clearance:.3f} mm")
                
                if clearance < 0.01:
                    print(f"   ❌ SAME RADIUS - Not a fit")
                    continue
                elif clearance > tolerance * 20:
                    print(f"   ❌ CLEARANCE TOO LARGE (>{tolerance*20:.1f} mm)")
                    continue
                else:
                    print(f"   ✓ Clearance OK")
                
                # Check axis alignment
                n1 = c1.normal if c1.normal is not None else np.array([0, 0, 1])
                n2 = c2.normal if c2.normal is not None else np.array([0, 0, 1])
                dot = abs(np.dot(n1, n2))
                
                print(f"   Axis alignment (dot product): {dot:.3f}")
                
                if dot < 0.95:
                    print(f"   ❌ AXES NOT PARALLEL (need ≥0.95)")
                    angle_deg = np.arccos(min(dot, 1.0)) * 180 / np.pi
                    print(f"      Angle between axes: {angle_deg:.1f}°")
                    continue
                else:
                    print(f"   ✓ Axes parallel")
                
                # Check perpendicular distance between axes
                vec_between = c2.position - c1.position
                perpendicular = vec_between - np.dot(vec_between, n1) * n1
                perp_dist = np.linalg.norm(perpendicular)
                
                print(f"   Perpendicular distance between axes: {perp_dist:.3f} mm")
                
                if perp_dist >= 2.0:
                    print(f"   ❌ AXES TOO FAR APART (need <2.0 mm)")
                    print(f"      The shaft and hole axes are not aligned")
                    continue
                else:
                    print(f"   ✓ Axes close together")
                
                print(f"\n   ✅ THIS PAIR SHOULD BE DETECTED AS CONTACT!")
                print(f"      If not detected, there may be a bug in the code.")
    
    # Run actual contact detection
    print("\n" + "-"*80)
    print("RUNNING CONTACT DETECTION:")
    print("-"*80)
    
    detector = ContactDetector(features, tolerance=tolerance)
    contacts = detector.detect_contacts()
    
    print(f"\nContacts detected: {len(contacts)}")
    
    if len(contacts) == 0:
        print("\n⚠️  NO CONTACTS DETECTED")
        print("\nPossible reasons:")
        print("  1. Cylindrical features not detected (check if assembly has cylinders)")
        print("  2. Axes not aligned (check axis alignment above)")
        print("  3. Clearance too large or too small")
        print("  4. Features are on the same part")
        print(f"\nTry:")
        print(f"  • Increase tolerance: python analyze_step.py {step_file} 5.0")
        print(f"  • Check that assembly has a shaft going through a hole")
        print(f"  • Verify the STEP file loads correctly in a CAD viewer")
    else:
        for c in contacts:
            print(f"  ✓ {c.contact_type}: Feature {c.feature1.id} <-> {c.feature2.id}")
            print(f"    Distance: {c.distance:.3f} mm")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_assembly.py <step_file> [tolerance]")
        print("\nExample:")
        print("  python diagnose_assembly.py Assembly_1.step")
        print("  python diagnose_assembly.py Assembly_1.step 5.0")
        sys.exit(1)
    
    step_file = sys.argv[1]
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    
    diagnose_assembly(step_file, tolerance)
