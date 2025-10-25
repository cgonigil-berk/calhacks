"""
Mating Interface Detector - Phase 2
Identifies shaft-hole interfaces across parts in STEP assemblies.
Critical for determining where tolerances are needed.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1


class CylinderType(Enum):
    """Classification of cylindrical features."""
    HOLE = "hole"           # Internal cylindrical feature (e.g., drilled hole)
    SHAFT = "shaft"         # External cylindrical feature (e.g., pin, boss)
    UNKNOWN = "unknown"     # Cannot determine from geometry alone


class FitType(Enum):
    """Type of fit between mating cylinders."""
    CLEARANCE = "clearance"         # Shaft smaller than hole (e.g., bolt in hole)
    TRANSITION = "transition"       # Tight fit, can be either
    INTERFERENCE = "interference"   # Shaft larger than hole (press fit)
    NO_FIT = "no_fit"              # Not mating


@dataclass
class CylindricalFeature:
    """
    Represents a cylindrical feature (hole or shaft) in a part.
    """
    part_id: int
    feature_id: int
    cylinder_type: CylinderType
    diameter: float                 # mm
    radius: float                   # mm
    axis_location: Tuple[float, float, float]  # (x, y, z) point on axis
    axis_direction: Tuple[float, float, float] # (dx, dy, dz) unit vector
    surface_area: float            # mm²
    
    # Derived properties for analysis
    @property
    def axis_vector(self) -> np.ndarray:
        """Return axis direction as numpy array."""
        return np.array(self.axis_direction)
    
    @property
    def location_vector(self) -> np.ndarray:
        """Return axis location as numpy array."""
        return np.array(self.axis_location)


@dataclass
class MatingInterface:
    """
    Represents a detected mating interface between two cylindrical features.
    """
    hole_feature: CylindricalFeature
    shaft_feature: CylindricalFeature
    clearance: float               # mm (hole_diameter - shaft_diameter)
    fit_type: FitType
    alignment_angle: float         # degrees (0° = perfectly aligned)
    axial_offset: float           # mm (distance between axes)
    priority: str                 # 'critical', 'important', 'standard'
    
    @property
    def recommended_tolerance_grade(self) -> str:
        """
        Recommend ISO 286 tolerance grade based on fit type.
        """
        if self.fit_type == FitType.CLEARANCE:
            if self.clearance < 0.1:
                return "H7/g6"  # Close clearance fit
            else:
                return "H11/h11"  # Loose clearance fit
        elif self.fit_type == FitType.TRANSITION:
            return "H7/k6"  # Transition fit
        elif self.fit_type == FitType.INTERFERENCE:
            return "H7/p6"  # Press fit
        return "H11/h11"  # Default


class MatingInterfaceDetector:
    """
    Detects mating interfaces (shaft-hole fits) in STEP assemblies.
    Critical for identifying where tolerances are needed.
    """
    
    def __init__(self, step_file_path: str):
        """
        Initialize detector with STEP file.
        
        Args:
            step_file_path: Path to STEP assembly file
        """
        self.step_file_path = step_file_path
        self.reader = STEPControl_Reader()
        self.shape: Optional[TopoDS_Shape] = None
        self.parts: List[TopoDS_Solid] = []
        self.cylindrical_features: List[CylindricalFeature] = []
        self.mating_interfaces: List[MatingInterface] = []
        
        # Detection thresholds (engineering judgment)
        self.ALIGNMENT_THRESHOLD_DEG = 5.0    # Max angle for "aligned" axes
        self.PROXIMITY_THRESHOLD_MM = 0.5     # Max offset for "mating"
        self.MIN_CLEARANCE_MM = -0.1          # Negative = interference
        self.MAX_CLEARANCE_MM = 2.0           # Beyond this = not mating
    
    def parse_step_file(self) -> bool:
        """Load and parse STEP file."""
        try:
            status = self.reader.ReadFile(self.step_file_path)
            if status != IFSelect_RetDone:
                raise ValueError(f"Failed to read STEP file")
            
            self.reader.TransferRoots()
            self.shape = self.reader.Shape()
            
            if self.shape.IsNull():
                raise ValueError("STEP file contains no valid geometry")
            
            # Extract all solid parts
            explorer = TopExp_Explorer(self.shape, TopAbs_SOLID)
            while explorer.More():
                self.parts.append(explorer.Current())
                explorer.Next()
            
            print(f"✓ Parsed STEP file: {len(self.parts)} parts found")
            return True
            
        except Exception as e:
            print(f"✗ Error parsing STEP file: {e}")
            return False
    
    def extract_all_cylindrical_features(self):
        """
        Extract all cylindrical features from all parts in the assembly.
        Initially marks all as UNKNOWN - will classify during mating detection.
        """
        self.cylindrical_features = []
        feature_id = 0
        
        for part_id, solid in enumerate(self.parts):
            face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
            
            part_cylinders = []
            
            while face_explorer.More():
                face = face_explorer.Current()
                surface = BRepAdaptor_Surface(face)
                
                if surface.GetType() == GeomAbs_Cylinder:
                    # Extract cylinder parameters
                    cylinder = surface.Cylinder()
                    radius = cylinder.Radius()
                    diameter = 2 * radius
                    
                    axis = cylinder.Axis()
                    location = (axis.Location().X(), 
                              axis.Location().Y(), 
                              axis.Location().Z())
                    direction = (axis.Direction().X(),
                               axis.Direction().Y(),
                               axis.Direction().Z())
                    
                    # Calculate surface area
                    props = GProp_GProps()
                    brepgprop_SurfaceProperties(face, props)
                    area = props.Mass()
                    
                    # Start as UNKNOWN - will classify based on mating relationships
                    feature = CylindricalFeature(
                        part_id=part_id,
                        feature_id=feature_id,
                        cylinder_type=CylinderType.UNKNOWN,
                        diameter=diameter,
                        radius=radius,
                        axis_location=location,
                        axis_direction=direction,
                        surface_area=area
                    )
                    
                    part_cylinders.append(feature)
                    feature_id += 1
                
                face_explorer.Next()
            
            self.cylindrical_features.extend(part_cylinders)
            print(f"  Part {part_id}: {len(part_cylinders)} cylindrical features")
        
        print(f"\n✓ Total cylindrical features: {len(self.cylindrical_features)}")
    
    def _classify_cylinder(self, face, solid, direction: Tuple) -> CylinderType:
        """
        Classify cylinder as hole or shaft based on geometric context.
        
        Strategy:
        - Holes: Internal features, face normal points inward
        - Shafts: External features, face normal points outward
        
        Args:
            face: The cylindrical face
            solid: The solid part containing this face
            direction: Axis direction
            
        Returns:
            CylinderType classification
        """
        # Check face orientation relative to solid
        # This is a simplified heuristic - in production would use more robust methods
        
        from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
        from OCC.Core.gp import gp_Pnt
        
        # Get a point on the cylindrical surface
        surface = BRepAdaptor_Surface(face)
        cylinder = surface.Cylinder()
        
        # Sample point slightly offset from cylinder axis (outward)
        axis_loc = cylinder.Location()
        axis_dir = cylinder.Axis().Direction()
        radius = cylinder.Radius()
        
        # Create a point slightly outside the cylinder
        test_point = gp_Pnt(
            axis_loc.X() + radius * 1.1,
            axis_loc.Y(),
            axis_loc.Z()
        )
        
        # Classify point relative to solid
        classifier = BRepClass3d_SolidClassifier(solid, test_point, 1e-6)
        state = classifier.State()
        
        # If point outside cylinder is also outside solid → external feature (shaft)
        # If point outside cylinder is inside solid → internal feature (hole)
        from OCC.Core.TopAbs import TopAbs_OUT, TopAbs_IN
        
        if state == TopAbs_OUT:
            return CylinderType.SHAFT
        elif state == TopAbs_IN:
            return CylinderType.HOLE
        else:
            return CylinderType.UNKNOWN
    
    def detect_mating_interfaces(self):
        """
        Detect mating interfaces between cylindrical features.
        Identifies shaft-hole pairs that are likely mating in the assembly.
        Classifies cylinders as hole/shaft based on diameter comparison.
        """
        self.mating_interfaces = []
        
        print(f"\nAnalyzing {len(self.cylindrical_features)} cylindrical features...")
        
        # Check all cylinder pairs from different parts
        for i, cyl1 in enumerate(self.cylindrical_features):
            for j, cyl2 in enumerate(self.cylindrical_features):
                if i >= j:
                    continue
                
                # Skip if same part (internal mating unlikely)
                if cyl1.part_id == cyl2.part_id:
                    continue
                
                # Check if they're spatially aligned and proximate
                is_mating, alignment_angle, axial_offset = self._check_alignment(cyl1, cyl2)
                
                if is_mating:
                    # Classify based on diameter: larger = hole, smaller = shaft
                    if cyl1.diameter > cyl2.diameter:
                        hole = cyl1
                        shaft = cyl2
                    else:
                        hole = cyl2
                        shaft = cyl1
                    
                    # Update classifications
                    hole.cylinder_type = CylinderType.HOLE
                    shaft.cylinder_type = CylinderType.SHAFT
                    
                    # Calculate clearance (positive = clearance, negative = interference)
                    clearance = hole.diameter - shaft.diameter
                    
                    # Determine fit type
                    fit_type = self._determine_fit_type(clearance, hole.diameter)
                    
                    # Determine priority based on fit tightness
                    priority = self._determine_priority(clearance, fit_type)
                    
                    interface = MatingInterface(
                        hole_feature=hole,
                        shaft_feature=shaft,
                        clearance=clearance,
                        fit_type=fit_type,
                        alignment_angle=alignment_angle,
                        axial_offset=axial_offset,
                        priority=priority
                    )
                    
                    self.mating_interfaces.append(interface)
                    
                    print(f"  ✓ Found mating: Part {hole.part_id} Ø{hole.diameter:.2f} hole " +
                          f"← Part {shaft.part_id} Ø{shaft.diameter:.2f} shaft " +
                          f"(clearance: {clearance:.3f}mm, {fit_type.value})")
        
        print(f"\n✓ Total mating interfaces detected: {len(self.mating_interfaces)}")
    
    def _check_alignment(self, hole: CylindricalFeature, 
                        shaft: CylindricalFeature) -> Tuple[bool, float, float]:
        """
        Check if hole and shaft axes are aligned (indicating potential mating).
        
        Returns:
            (is_aligned, alignment_angle_deg, axial_offset_mm)
        """
        # Check axis alignment (angle between direction vectors)
        hole_axis = hole.axis_vector
        shaft_axis = shaft.axis_vector
        
        # Normalize
        hole_axis = hole_axis / np.linalg.norm(hole_axis)
        shaft_axis = shaft_axis / np.linalg.norm(shaft_axis)
        
        # Angle between axes (account for opposite directions)
        dot_product = np.abs(np.dot(hole_axis, shaft_axis))
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Numerical stability
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Check if axes are nearly parallel
        if angle_deg > self.ALIGNMENT_THRESHOLD_DEG:
            return False, angle_deg, float('inf')
        
        # Check spatial proximity (distance between axes)
        # Using point-to-line distance formula
        hole_loc = hole.location_vector
        shaft_loc = shaft.location_vector
        
        # Vector from shaft to hole location
        loc_diff = hole_loc - shaft_loc
        
        # Distance from hole location to shaft axis
        # d = ||(loc_diff) - ((loc_diff · shaft_axis) * shaft_axis)||
        projection = np.dot(loc_diff, shaft_axis) * shaft_axis
        perpendicular = loc_diff - projection
        axial_offset = np.linalg.norm(perpendicular)
        
        # Check if axes are close enough to be mating
        is_mating = axial_offset < self.PROXIMITY_THRESHOLD_MM
        
        return is_mating, angle_deg, axial_offset
    
    def _determine_fit_type(self, clearance: float, hole_diameter: float) -> FitType:
        """
        Determine fit type based on clearance value.
        
        Per ISO 286 standards:
        - Clearance fit: positive clearance
        - Transition fit: small clearance or small interference
        - Interference fit: negative clearance (shaft > hole)
        
        Args:
            clearance: hole_diameter - shaft_diameter (mm)
            hole_diameter: Nominal hole diameter (mm)
        """
        # Calculate relative clearance (as % of diameter)
        relative_clearance = (clearance / hole_diameter) * 100
        
        if clearance < self.MIN_CLEARANCE_MM:
            # Significant interference
            return FitType.INTERFERENCE
        elif clearance < 0.05:
            # Very tight fit, could be transition or light interference
            return FitType.TRANSITION
        elif clearance > self.MAX_CLEARANCE_MM:
            # Too much clearance, probably not a designed fit
            return FitType.NO_FIT
        else:
            # Normal clearance fit
            return FitType.CLEARANCE
    
    def _determine_priority(self, clearance: float, fit_type: FitType) -> str:
        """
        Determine tolerance priority based on clearance and fit type.
        
        Critical: Tight fits, precision assembly
        Important: Standard fits with moderate clearance
        Standard: Loose fits, non-functional clearances
        """
        if fit_type == FitType.INTERFERENCE:
            return "critical"
        elif fit_type == FitType.TRANSITION:
            return "critical"
        elif abs(clearance) < 0.2:
            return "important"
        else:
            return "standard"
    
    def generate_tolerance_report(self) -> Dict:
        """
        Generate structured tolerance recommendations for all mating interfaces.
        
        Returns:
            Dictionary containing tolerance specifications for each interface
        """
        report = {
            "assembly": self.step_file_path,
            "total_parts": len(self.parts),
            "total_cylindrical_features": len(self.cylindrical_features),
            "mating_interfaces": len(self.mating_interfaces),
            "recommendations": []
        }
        
        for i, interface in enumerate(self.mating_interfaces, 1):
            hole = interface.hole_feature
            shaft = interface.shaft_feature
            
            recommendation = {
                "interface_id": i,
                "priority": interface.priority,
                "description": f"Part {hole.part_id} hole mates with Part {shaft.part_id} shaft",
                
                # Hole specifications
                "hole": {
                    "part_id": hole.part_id,
                    "nominal_diameter": round(hole.diameter, 3),
                    "recommended_tolerance": self._recommend_hole_tolerance(interface),
                    "gdt_callouts": self._recommend_hole_gdt(interface)
                },
                
                # Shaft specifications
                "shaft": {
                    "part_id": shaft.part_id,
                    "nominal_diameter": round(shaft.diameter, 3),
                    "recommended_tolerance": self._recommend_shaft_tolerance(interface),
                    "gdt_callouts": self._recommend_shaft_gdt(interface)
                },
                
                # Fit analysis
                "fit_analysis": {
                    "clearance": round(interface.clearance, 4),
                    "fit_type": interface.fit_type.value,
                    "iso_286_grade": interface.recommended_tolerance_grade,
                    "alignment_angle": round(interface.alignment_angle, 2),
                    "axial_offset": round(interface.axial_offset, 4)
                },
                
                # Manufacturing notes
                "manufacturing_notes": self._generate_manufacturing_notes(interface)
            }
            
            report["recommendations"].append(recommendation)
        
        return report
    
    def _recommend_hole_tolerance(self, interface: MatingInterface) -> str:
        """Recommend dimensional tolerance for hole."""
        fit_grade = interface.recommended_tolerance_grade
        hole_tol = fit_grade.split('/')[0]  # e.g., "H7" from "H7/g6"
        
        # Typical values per ISO 286 (simplified)
        diameter = interface.hole_feature.diameter
        
        if "H11" in hole_tol:
            tol = 0.11 if diameter < 50 else 0.13
        elif "H7" in hole_tol:
            tol = 0.025 if diameter < 50 else 0.030
        else:
            tol = 0.05
        
        return f"+{tol:.3f}/-0.000"
    
    def _recommend_shaft_tolerance(self, interface: MatingInterface) -> str:
        """Recommend dimensional tolerance for shaft."""
        fit_grade = interface.recommended_tolerance_grade
        shaft_tol = fit_grade.split('/')[1]  # e.g., "g6" from "H7/g6"
        
        diameter = interface.shaft_feature.diameter
        
        if "h11" in shaft_tol:
            tol = 0.11 if diameter < 50 else 0.13
            return f"+0.000/-{tol:.3f}"
        elif "g6" in shaft_tol:
            return f"-0.005/-0.018"
        elif "k6" in shaft_tol:
            return f"+0.006/-0.006"
        elif "p6" in shaft_tol:
            return f"+0.018/+0.009"
        else:
            return f"+0.000/-0.050"
    
    def _recommend_hole_gdt(self, interface: MatingInterface) -> List[str]:
        """
        Recommend GD&T callouts for hole.
        Per ASME Y14.5-2018 principles.
        """
        callouts = []
        
        # Position tolerance for hole location
        # Tighter for critical fits
        if interface.priority == "critical":
            pos_tol = 0.05
        elif interface.priority == "important":
            pos_tol = 0.1
        else:
            pos_tol = 0.2
        
        callouts.append(f"Position Ø{pos_tol:.2f} [A|B|C]")
        
        # Perpendicularity for hole axis (if vertical mounting)
        if abs(interface.hole_feature.axis_direction[2]) > 0.9:  # Nearly vertical
            perp_tol = pos_tol * 0.5
            callouts.append(f"Perpendicularity {perp_tol:.3f} relative to [A]")
        
        # Cylindricity for tight fits
        if interface.fit_type in [FitType.INTERFERENCE, FitType.TRANSITION]:
            cyl_tol = 0.01
            callouts.append(f"Cylindricity {cyl_tol:.3f}")
        
        return callouts
    
    def _recommend_shaft_gdt(self, interface: MatingInterface) -> List[str]:
        """Recommend GD&T callouts for shaft."""
        callouts = []
        
        # Position tolerance for shaft
        if interface.priority == "critical":
            pos_tol = 0.05
        else:
            pos_tol = 0.1
        
        callouts.append(f"Position Ø{pos_tol:.2f} [A|B|C]")
        
        # Cylindricity for mating surface
        if interface.fit_type in [FitType.INTERFERENCE, FitType.TRANSITION]:
            cyl_tol = 0.005
            callouts.append(f"Cylindricity {cyl_tol:.3f}")
        
        # Straightness for long shafts
        shaft_length_estimate = interface.shaft_feature.surface_area / (np.pi * interface.shaft_feature.diameter)
        if shaft_length_estimate > 50:  # mm
            callouts.append(f"Straightness {0.02:.3f} per 100mm")
        
        return callouts
    
    def _generate_manufacturing_notes(self, interface: MatingInterface) -> List[str]:
        """Generate manufacturing guidance notes."""
        notes = []
        
        if interface.fit_type == FitType.CLEARANCE:
            notes.append("Standard machining tolerances sufficient")
            notes.append("Recommended: Drilling + reaming for holes")
        elif interface.fit_type == FitType.TRANSITION:
            notes.append("Precision machining required")
            notes.append("Consider honing or grinding for final size")
        elif interface.fit_type == FitType.INTERFERENCE:
            notes.append("Press fit - requires precision grinding")
            notes.append("Assembly may require heating/cooling")
            notes.append("Verify material properties for stress")
        
        # Clearance-specific notes
        if interface.clearance < 0.05:
            notes.append(f"Very tight clearance ({interface.clearance:.3f}mm) - critical tolerance control")
        
        return notes
    
    def print_tolerance_report(self):
        """Print human-readable tolerance report."""
        report = self.generate_tolerance_report()
        
        print("\n" + "="*80)
        print("TOLERANCE SPECIFICATION REPORT")
        print("="*80)
        print(f"Assembly: {report['assembly']}")
        print(f"Total Parts: {report['total_parts']}")
        print(f"Cylindrical Features: {report['total_cylindrical_features']}")
        print(f"Mating Interfaces Detected: {report['mating_interfaces']}")
        print("="*80)
        
        for rec in report["recommendations"]:
            print(f"\n{'='*80}")
            print(f"INTERFACE {rec['interface_id']}: {rec['description']}")
            print(f"Priority: {rec['priority'].upper()}")
            print(f"{'='*80}")
            
            print(f"\n  HOLE (Part {rec['hole']['part_id']}):")
            print(f"    Nominal: Ø{rec['hole']['nominal_diameter']:.3f} mm")
            print(f"    Tolerance: {rec['hole']['recommended_tolerance']}")
            print(f"    GD&T Callouts:")
            for callout in rec['hole']['gdt_callouts']:
                print(f"      • {callout}")
            
            print(f"\n  SHAFT (Part {rec['shaft']['part_id']}):")
            print(f"    Nominal: Ø{rec['shaft']['nominal_diameter']:.3f} mm")
            print(f"    Tolerance: {rec['shaft']['recommended_tolerance']}")
            print(f"    GD&T Callouts:")
            for callout in rec['shaft']['gdt_callouts']:
                print(f"      • {callout}")
            
            print(f"\n  FIT ANALYSIS:")
            print(f"    Clearance: {rec['fit_analysis']['clearance']:.4f} mm")
            print(f"    Fit Type: {rec['fit_analysis']['fit_type']}")
            print(f"    ISO 286 Grade: {rec['fit_analysis']['iso_286_grade']}")
            print(f"    Alignment: {rec['fit_analysis']['alignment_angle']:.2f}°")
            
            print(f"\n  MANUFACTURING NOTES:")
            for note in rec['manufacturing_notes']:
                print(f"    • {note}")
        
        print("\n" + "="*80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of mating interface detector."""
    
    import sys
    
    if len(sys.argv) < 2:
        step_file = "Assembly_1.step"
        print(f"Using default file: {step_file}")
    else:
        step_file = sys.argv[1]
    
    # Initialize detector
    detector = MatingInterfaceDetector(step_file)
    
    # Parse STEP file
    if not detector.parse_step_file():
        return
    
    # Extract all cylindrical features
    print("\n" + "="*80)
    print("EXTRACTING CYLINDRICAL FEATURES")
    print("="*80)
    detector.extract_all_cylindrical_features()
    
    # Detect mating interfaces
    print("\n" + "="*80)
    print("DETECTING MATING INTERFACES")
    print("="*80)
    detector.detect_mating_interfaces()
    
    # Generate and print tolerance report
    detector.print_tolerance_report()


if __name__ == "__main__":
    main()