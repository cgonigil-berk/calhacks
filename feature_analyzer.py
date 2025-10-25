"""
Feature Analyzer Module
Identifies geometric features from STEP assembly files
"""

import cadquery as cq
from cadquery import exporters
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FeatureType(Enum):
    """Types of geometric features that can be identified"""
    HOLE = "Hole"
    BOSS = "Boss"
    SLOT = "Slot"
    FACE = "Flat Face"
    CYLINDER = "Cylindrical Surface"
    EDGE = "Edge"
    UNKNOWN = "Unknown"


@dataclass
class Feature:
    """Represents a geometric feature"""
    id: int
    part_id: int
    feature_type: FeatureType
    position: np.ndarray
    normal: Optional[np.ndarray]
    dimensions: Dict[str, float]
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    
    def __repr__(self):
        return f"Feature({self.feature_type.value}, Part {self.part_id}, ID {self.id})"


class FeatureAnalyzer:
    """Analyzes STEP files to identify geometric features"""
    
    def __init__(self, step_file_path: str):
        self.step_file_path = step_file_path
        self.parts = []
        self.features = []
        self._load_step_file()
    
    def _load_step_file(self):
        """Load STEP file and extract parts"""
        try:
            # Import using OCP directly for better assembly support
            from OCP.STEPControl import STEPControl_Reader
            from OCP.IFSelect import IFSelect_RetDone
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID
            
            reader = STEPControl_Reader()
            status = reader.ReadFile(self.step_file_path)
            
            if status != IFSelect_RetDone:
                raise Exception("Error reading STEP file")
            
            # Transfer all roots
            reader.TransferRoots()
            
            # Get the shape
            shape = reader.OneShape()
            
            # Extract individual solids from the shape
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            solids = []
            
            while explorer.More():
                solid = explorer.Current()
                solids.append(solid)
                explorer.Next()
            
            # If we found multiple solids, treat each as a separate part
            if len(solids) > 1:
                for solid in solids:
                    self.parts.append(cq.Workplane().add(solid))
            elif len(solids) == 1:
                # Single solid
                self.parts.append(cq.Workplane().add(solids[0]))
            else:
                # Fallback: use the whole shape
                self.parts.append(cq.Workplane().add(shape))
            
            print(f"Loaded {len(self.parts)} part(s) from STEP file")
                
        except Exception as e:
            print(f"Error loading STEP file: {e}")
            raise Exception(f"Could not load STEP file: {e}")
    
    def identify_features(self) -> List[Feature]:
        """Identify all features in all parts"""
        self.features = []
        feature_id = 0
        
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        
        for part_idx, part in enumerate(self.parts):
            try:
                # Get the underlying OCC shape
                if hasattr(part, 'val'):
                    shape_obj = part.val()
                    # The shape_obj itself is already an OCC object
                    if hasattr(shape_obj, 'wrapped'):
                        shape = shape_obj.wrapped
                    else:
                        shape = shape_obj
                else:
                    shape = part
                
                # Explore faces
                explorer = TopExp_Explorer(shape, TopAbs_FACE)
                face_count = 0
                
                while explorer.More():
                    occ_face = explorer.Current()
                    face_count += 1
                    
                    # Wrap in CadQuery Face for easier manipulation
                    try:
                        cq_face = cq.Face(occ_face)
                        feature = self._analyze_face(cq_face, part_idx, feature_id)
                        if feature:
                            self.features.append(feature)
                            feature_id += 1
                    except Exception as e:
                        print(f"Error analyzing face {face_count} in part {part_idx}: {e}")
                    
                    explorer.Next()
                
                print(f"Part {part_idx}: analyzed {face_count} faces, identified {len([f for f in self.features if f.part_id == part_idx])} features")
                
            except Exception as e:
                print(f"Error analyzing part {part_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Total features identified: {len(self.features)}")
        return self.features
    
    def _analyze_face(self, face, part_id: int, feature_id: int) -> Optional[Feature]:
        """Analyze a face to determine its feature type"""
        try:
            # Get face properties using CadQuery methods
            center = face.Center()
            bbox = face.BoundingBox()
            
            # Convert to numpy arrays
            position = np.array([center.x, center.y, center.z])
            bbox_tuple = (
                (bbox.xmin, bbox.ymin, bbox.zmin),
                (bbox.xmax, bbox.ymax, bbox.zmax)
            )
            
            # Get the underlying OCC surface
            from OCP.GeomAbs import GeomAbs_SurfaceType
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            
            surface_adaptor = BRepAdaptor_Surface(face.wrapped)
            surface_type = surface_adaptor.GetType()
            
            # Determine feature type based on surface geometry
            if surface_type == GeomAbs_SurfaceType.GeomAbs_Plane:
                # Get plane normal
                try:
                    normal_vec = face.normalAt()
                    normal = np.array([normal_vec.x, normal_vec.y, normal_vec.z])
                except:
                    normal = np.array([0, 0, 1])
                
                area = face.Area()
                dimensions = {
                    "area": area,
                    "width": bbox.xmax - bbox.xmin,
                    "height": bbox.ymax - bbox.ymin
                }
                
                return Feature(
                    id=feature_id,
                    part_id=part_id,
                    feature_type=FeatureType.FACE,
                    position=position,
                    normal=normal,
                    dimensions=dimensions,
                    bounding_box=bbox_tuple
                )
            
            elif surface_type == GeomAbs_SurfaceType.GeomAbs_Cylinder:
                try:
                    # Cylindrical surface
                    cylinder = surface_adaptor.Cylinder()
                    radius = cylinder.Radius()
                    dimensions = {"radius": radius}
                    
                    # Get axis direction
                    try:
                        axis = cylinder.Axis()
                        normal = np.array([axis.Direction().X(), 
                                         axis.Direction().Y(), 
                                         axis.Direction().Z()])
                    except:
                        normal = np.array([0, 0, 1])
                    
                    return Feature(
                        id=feature_id,
                        part_id=part_id,
                        feature_type=FeatureType.CYLINDER,
                        position=position,
                        normal=normal,
                        dimensions=dimensions,
                        bounding_box=bbox_tuple
                    )
                except Exception as e:
                    print(f"Error analyzing cylinder: {e}")
            
        except Exception as e:
            print(f"Error analyzing face: {e}")
        
        return None
    
    def _analyze_hole(self, face, part_id: int, feature_id: int) -> Optional[Feature]:
        """Analyze a circular face to determine if it's a hole"""
        try:
            center = face.Center()
            position = np.array([center.x, center.y, center.z])
            bbox = face.BoundingBox()
            
            # Estimate hole diameter from bounding box
            diameter = max(bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin)
            
            try:
                normal_vec = face.normalAt()
                normal = np.array([normal_vec.x, normal_vec.y, normal_vec.z])
            except:
                normal = np.array([0, 0, 1])
            
            dimensions = {
                "diameter": diameter,
                "radius": diameter / 2
            }
            
            bbox_tuple = (
                (bbox.xmin, bbox.ymin, bbox.zmin),
                (bbox.xmax, bbox.ymax, bbox.zmax)
            )
            
            return Feature(
                id=feature_id,
                part_id=part_id,
                feature_type=FeatureType.HOLE,
                position=position,
                normal=normal,
                dimensions=dimensions,
                bounding_box=bbox_tuple
            )
            
        except Exception as e:
            print(f"Error analyzing hole: {e}")
        
        return None
    
    def get_part_count(self) -> int:
        """Return the number of parts in the assembly"""
        return len(self.parts)
    
    def get_feature_count(self) -> int:
        """Return the total number of features identified"""
        return len(self.features)
