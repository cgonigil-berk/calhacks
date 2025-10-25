"""
Contact Detector Module
Identifies when features are in contact with each other
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from feature_analyzer import Feature, FeatureType


@dataclass
class Contact:
    """Represents contact between two features"""
    feature1: Feature
    feature2: Feature
    contact_type: str
    distance: float
    contact_area: float
    
    def __repr__(self):
        return f"Contact({self.feature1.id} <-> {self.feature2.id}, {self.contact_type}, dist={self.distance:.4f})"


class ContactDetector:
    """Detects contact between features"""
    
    def __init__(self, features: List[Feature], tolerance: float = 0.01):
        """
        Initialize contact detector
        
        Args:
            features: List of features to analyze
            tolerance: Distance threshold for determining contact (default 0.01 mm)
        """
        self.features = features
        self.tolerance = tolerance
        self.contacts = []
    
    def detect_contacts(self) -> List[Contact]:
        """Detect all contacts between features"""
        self.contacts = []
        
        # Check all pairs of features
        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                if i >= j:  # Skip duplicate pairs and self-comparison
                    continue
                
                # Skip features from the same part (internal features)
                if feature1.part_id == feature2.part_id:
                    continue
                
                contact = self._check_contact(feature1, feature2)
                if contact:
                    self.contacts.append(contact)
        
        return self.contacts
    
    def _check_contact(self, feature1: Feature, feature2: Feature) -> Contact:
        """Check if two features are in actual physical contact"""
        
        # Pre-filter: Check if bounding boxes are close enough to potentially contact
        if not self._bounding_boxes_close(feature1, feature2):
            return None
        
        type1 = feature1.feature_type
        type2 = feature2.feature_type
        
        # CASE 1: Two coaxial cylinders (shaft in hole or interference fit)
        # This works for both when we correctly identify holes and when we don't
        if (type1 == FeatureType.CYLINDER or type1 == FeatureType.HOLE) and \
           (type2 == FeatureType.CYLINDER or type2 == FeatureType.HOLE):
            
            r1 = feature1.dimensions.get("radius", 0)
            r2 = feature2.dimensions.get("radius", 0)
            
            # Check if axes are coaxial (parallel and close)
            if self._cylinders_coaxial(feature1, feature2):
                clearance = abs(r1 - r2)
                
                # Accept contacts with reasonable clearance
                # Scale tolerance with radius size for better detection
                max_clearance = max(self.tolerance * 20, min(r1, r2) * 0.1)
                
                if clearance <= max_clearance:
                    # Determine contact type based on clearance
                    if clearance < 0.1:  # Very tight fit or interference
                        contact_type = "Interference/Transition Fit"
                    else:
                        contact_type = "Clearance/Interference Fit"
                    
                    # Calculate contact length from overlapping bounding boxes
                    contact_length = self._calculate_cylinder_overlap_length(feature1, feature2)
                    contact_area = 2 * np.pi * min(r1, r2) * contact_length
                    
                    return Contact(
                        feature1=feature1,
                        feature2=feature2,
                        contact_type=contact_type,
                        distance=clearance,
                        contact_area=contact_area
                    )
        
        # CASE 2: Two cylinders (side-by-side, not coaxial)
        elif (type1 == FeatureType.CYLINDER and type2 == FeatureType.CYLINDER):
            if self._cylinders_touching(feature1, feature2):
                distance = np.linalg.norm(feature1.position - feature2.position)
                r1 = feature1.dimensions.get("radius", 0)
                r2 = feature2.dimensions.get("radius", 0)
                contact_area = min(r1, r2) * 10
                
                return Contact(
                    feature1=feature1,
                    feature2=feature2,
                    contact_type="Line Contact",
                    distance=distance,
                    contact_area=contact_area
                )
        
        # CASE 3: Face-to-Face contact (must be opposite normals and overlapping)
        elif type1 == FeatureType.FACE and type2 == FeatureType.FACE:
            if self._faces_touching(feature1, feature2):
                # Calculate actual overlapping area
                contact_area = self._calculate_face_overlap_area(feature1, feature2)
                
                if contact_area > 0.1:  # Minimum contact area threshold (0.1 mm²)
                    distance = self._distance_between_faces(feature1, feature2)
                    
                    # Determine contact type based on normals
                    dot_product = np.dot(feature1.normal, feature2.normal)
                    if dot_product < -0.8:  # Opposite normals
                        contact_type = "Mating Surfaces"
                    elif abs(dot_product) < 0.2:  # Perpendicular
                        contact_type = "Perpendicular Contact"
                    else:
                        contact_type = "Surface Contact"
                    
                    return Contact(
                        feature1=feature1,
                        feature2=feature2,
                        contact_type=contact_type,
                        distance=distance,
                        contact_area=contact_area
                    )
        
        # CASE 4: Cylinder touching a face
        elif (type1 == FeatureType.CYLINDER and type2 == FeatureType.FACE) or \
             (type1 == FeatureType.FACE and type2 == FeatureType.CYLINDER):
            
            if self._cylinder_touching_face(feature1, feature2):
                distance = self._distance_between_features(feature1, feature2)
                
                # Only accept if distance is within tolerance
                if distance <= self.tolerance * 2:
                    contact_area = 5.0  # Estimated edge contact area
                    
                    return Contact(
                        feature1=feature1,
                        feature2=feature2,
                        contact_type="Edge Contact",
                        distance=distance,
                        contact_area=contact_area
                    )
        
        return None
    
    def _cylinders_coaxial(self, cyl1: Feature, cyl2: Feature) -> bool:
        """Check if two cylinders are coaxial (same axis)"""
        
        pos1, pos2 = cyl1.position, cyl2.position
        normal1 = cyl1.normal if cyl1.normal is not None else np.array([0, 0, 1])
        normal2 = cyl2.normal if cyl2.normal is not None else np.array([0, 0, 1])
        
        # Normalize normals
        normal1 = normal1 / np.linalg.norm(normal1)
        normal2 = normal2 / np.linalg.norm(normal2)
        
        # Check if axes are parallel (or anti-parallel)
        dot_product = abs(np.dot(normal1, normal2))
        if dot_product < 0.98:  # Not parallel enough (within ~11 degrees)
            return False
        
        # Check if axes are close (perpendicular distance between axes)
        vec_between = pos2 - pos1
        
        # Project onto perpendicular plane to get radial offset
        perpendicular = vec_between - np.dot(vec_between, normal1) * normal1
        perp_distance = np.linalg.norm(perpendicular)
        
        # Get radii to scale the tolerance
        r1 = cyl1.dimensions.get("radius", 0)
        r2 = cyl2.dimensions.get("radius", 0)
        max_radius = max(r1, r2)
        
        # Axes must be very close - scale with cylinder size
        # Allow up to 5% of the larger radius or 2mm, whichever is smaller
        max_offset = min(max_radius * 0.05, 2.0)
        
        if perp_distance <= max_offset:
            return True
        
        return False
    
    def _cylinders_touching(self, cyl1: Feature, cyl2: Feature) -> bool:
        """Check if two cylinders are touching"""
        
        pos1, pos2 = cyl1.position, cyl2.position
        r1 = cyl1.dimensions.get("radius", 0)
        r2 = cyl2.dimensions.get("radius", 0)
        
        distance = np.linalg.norm(pos1 - pos2)
        
        # Check if surfaces touch (distance between centers ≈ sum of radii)
        if abs(distance - (r1 + r2)) < self.tolerance:
            return True
        
        return False
    
    def _faces_touching(self, face1: Feature, face2: Feature) -> bool:
        """Check if two faces are actually touching (not just coplanar)"""
        
        # Faces must have opposite normals to be touching
        if face1.normal is None or face2.normal is None:
            return False
        
        # Normalize normals
        normal1 = face1.normal / np.linalg.norm(face1.normal)
        normal2 = face2.normal / np.linalg.norm(face2.normal)
        
        dot_product = np.dot(normal1, normal2)
        
        # Must be opposite directions (dot product close to -1)
        if dot_product > -0.7:
            return False
        
        # Calculate distance between faces along normal direction
        vec_between = face2.position - face1.position
        distance_along_normal = abs(np.dot(vec_between, normal1))
        
        # Faces must be very close (within tolerance)
        if distance_along_normal > self.tolerance:
            return False
        
        # Check if bounding boxes overlap in 3D space first (quick rejection)
        bbox1_min, bbox1_max = face1.bounding_box
        bbox2_min, bbox2_max = face2.bounding_box
        
        # Add tolerance margin for bbox check
        margin = self.tolerance
        if not self._bboxes_overlap_3d(bbox1_min, bbox1_max, bbox2_min, bbox2_max, margin):
            return False
        
        # Check if faces overlap when projected onto a common plane
        # Use the plane of face1 as reference
        overlap_area = self._calculate_projected_overlap(face1, face2)
        
        # Require minimum overlap area
        if overlap_area < 0.1:  # 0.1 mm²
            return False
        
        return True
    
    def _get_bbox_corners(self, bbox_min: Tuple[float, float, float], 
                          bbox_max: Tuple[float, float, float]) -> List[np.ndarray]:
        """Get all 8 corners of a bounding box"""
        corners = []
        for x in [bbox_min[0], bbox_max[0]]:
            for y in [bbox_min[1], bbox_max[1]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    corners.append(np.array([x, y, z]))
        return corners
    
    def _calculate_face_overlap_area(self, face1: Feature, face2: Feature) -> float:
        """Calculate the actual overlapping area between two faces"""
        
        # Simplified: use minimum of the two face areas as approximation
        # A proper implementation would project and calculate intersection polygon
        area1 = face1.dimensions.get("area", 0)
        area2 = face2.dimensions.get("area", 0)
        
        # Check dimensional overlap
        bbox1_min, bbox1_max = face1.bounding_box
        bbox2_min, bbox2_max = face2.bounding_box
        
        # Calculate overlap dimensions
        overlap_x = max(0, min(bbox1_max[0], bbox2_max[0]) - max(bbox1_min[0], bbox2_min[0]))
        overlap_y = max(0, min(bbox1_max[1], bbox2_max[1]) - max(bbox1_min[1], bbox2_min[1]))
        overlap_z = max(0, min(bbox1_max[2], bbox2_max[2]) - max(bbox1_min[2], bbox2_min[2]))
        
        # Approximate overlap area (use two largest dimensions)
        dims = sorted([overlap_x, overlap_y, overlap_z], reverse=True)
        overlap_area = dims[0] * dims[1]
        
        # Return minimum of estimated overlap and actual face areas
        return min(overlap_area, area1, area2)
    
    def _distance_between_faces(self, face1: Feature, face2: Feature) -> float:
        """Calculate distance between two faces along their normal"""
        vec_between = face2.position - face1.position
        return abs(np.dot(vec_between, face1.normal))
    
    def _cylinder_touching_face(self, feat1: Feature, feat2: Feature) -> bool:
        """Check if a cylinder is touching a face with comprehensive geometric validation"""
        
        if feat1.feature_type == FeatureType.CYLINDER or feat1.feature_type == FeatureType.HOLE:
            cylinder, face = feat1, feat2
        else:
            cylinder, face = feat2, feat1
        
        if face.normal is None:
            return False
        
        # Get cylinder properties
        cyl_pos = cylinder.position
        cyl_axis = cylinder.normal if cylinder.normal is not None else np.array([0, 0, 1])
        cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
        r = cylinder.dimensions.get("radius", 0)
        
        # Get face properties
        face_pos = face.position
        face_normal = face.normal / np.linalg.norm(face.normal)
        
        # Calculate the closest point on the cylinder axis to the face plane
        # This helps determine if the cylinder actually intersects the face region
        vec_to_face = face_pos - cyl_pos
        
        # Distance from cylinder center to face plane
        dist_center_to_plane = np.dot(vec_to_face, face_normal)
        
        # Project cylinder center onto face plane
        closest_point_on_axis = cyl_pos + dist_center_to_plane * face_normal
        
        # Calculate perpendicular distance from cylinder axis to face plane
        # This tells us the orientation of the cylinder relative to the face
        axis_face_angle = abs(np.dot(cyl_axis, face_normal))
        
        # Case 1: Cylinder axis perpendicular to face (end of cylinder touching face)
        if axis_face_angle > 0.9:  # Nearly perpendicular
            # Distance from cylinder center to plane should be close to 0
            if abs(dist_center_to_plane) <= self.tolerance:
                # Check if cylinder center projects within face bounds
                if self._point_near_face_bounds(closest_point_on_axis, face):
                    return True
        
        # Case 2: Cylinder axis parallel or at angle to face (side of cylinder touching)
        else:
            # Distance from cylinder surface to plane
            # For a cylinder at an angle, we need to check if any point on the cylinder
            # surface is within tolerance of the face plane
            
            # The closest surface point to the plane is at distance |dist_center_to_plane| - r
            surface_to_plane_dist = abs(abs(dist_center_to_plane) - r)
            
            if surface_to_plane_dist <= self.tolerance:
                # Check if the contact region overlaps with the face bounds
                # Project cylinder axis onto face plane to find contact line
                if self._cylinder_axis_intersects_face_bounds(cylinder, face):
                    return True
        
        return False
    
    def _distance_between_features(self, feat1: Feature, feat2: Feature) -> float:
        """Calculate distance between two features"""
        return np.linalg.norm(feat1.position - feat2.position)
    
    def get_contact_count(self) -> int:
        """Return the number of contacts detected"""
        return len(self.contacts)
    
    def get_contacts_by_type(self) -> Dict[str, List[Contact]]:
        """Group contacts by type"""
        contacts_by_type = {}
        for contact in self.contacts:
            if contact.contact_type not in contacts_by_type:
                contacts_by_type[contact.contact_type] = []
            contacts_by_type[contact.contact_type].append(contact)
        return contacts_by_type
    
    def _bounding_boxes_close(self, feat1: Feature, feat2: Feature) -> bool:
        """Check if two features' bounding boxes are close enough to potentially contact"""
        bbox1_min, bbox1_max = feat1.bounding_box
        bbox2_min, bbox2_max = feat2.bounding_box
        
        # Expand bounding boxes by tolerance for proximity check
        margin = self.tolerance * 10  # Allow some margin for detection
        
        return self._bboxes_overlap_3d(bbox1_min, bbox1_max, bbox2_min, bbox2_max, margin)
    
    def _bboxes_overlap_3d(self, bbox1_min: Tuple[float, float, float], 
                           bbox1_max: Tuple[float, float, float],
                           bbox2_min: Tuple[float, float, float], 
                           bbox2_max: Tuple[float, float, float],
                           margin: float = 0.0) -> bool:
        """Check if two 3D bounding boxes overlap (with optional margin)"""
        # Check overlap in each dimension
        for i in range(3):
            # No overlap if one box is completely to the left/right of the other
            if bbox1_max[i] + margin < bbox2_min[i] or bbox2_max[i] + margin < bbox1_min[i]:
                return False
        return True
    
    def _calculate_cylinder_overlap_length(self, cyl1: Feature, cyl2: Feature) -> float:
        """Calculate the overlapping length of two coaxial cylinders along their axis"""
        bbox1_min, bbox1_max = cyl1.bounding_box
        bbox2_min, bbox2_max = cyl2.bounding_box
        
        # Determine the axis direction (which dimension varies most)
        normal = cyl1.normal if cyl1.normal is not None else np.array([0, 0, 1])
        
        # Find the dominant axis
        abs_normal = np.abs(normal)
        axis_idx = np.argmax(abs_normal)
        
        # Calculate overlap along that axis
        min1, max1 = bbox1_min[axis_idx], bbox1_max[axis_idx]
        min2, max2 = bbox2_min[axis_idx], bbox2_max[axis_idx]
        
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        overlap_length = max(0, overlap_end - overlap_start)
        
        # Default to 20mm if calculation fails
        return overlap_length if overlap_length > 0 else 20.0
    
    def _calculate_projected_overlap(self, face1: Feature, face2: Feature) -> float:
        """Calculate the overlapping area of two faces when projected onto a common plane"""
        # Simplified: use bounding box overlap as approximation
        bbox1_min, bbox1_max = face1.bounding_box
        bbox2_min, bbox2_max = face2.bounding_box
        
        # Calculate overlap in each dimension
        overlap_x = max(0, min(bbox1_max[0], bbox2_max[0]) - max(bbox1_min[0], bbox2_min[0]))
        overlap_y = max(0, min(bbox1_max[1], bbox2_max[1]) - max(bbox1_min[1], bbox2_min[1]))
        overlap_z = max(0, min(bbox1_max[2], bbox2_max[2]) - max(bbox1_min[2], bbox2_min[2]))
        
        # Use the two largest dimensions for area
        dims = sorted([overlap_x, overlap_y, overlap_z], reverse=True)
        return dims[0] * dims[1]
    
    def _point_near_face_bounds(self, point: np.ndarray, face: Feature) -> bool:
        """Check if a point is within or near the face's bounding box"""
        bbox_min, bbox_max = face.bounding_box
        margin = self.tolerance * 2
        
        for i in range(3):
            if point[i] < bbox_min[i] - margin or point[i] > bbox_max[i] + margin:
                return False
        return True
    
    def _cylinder_axis_intersects_face_bounds(self, cylinder: Feature, face: Feature) -> bool:
        """Check if the cylinder axis line intersects the face's bounding region"""
        # Get cylinder properties
        cyl_pos = cylinder.position
        cyl_axis = cylinder.normal if cylinder.normal is not None else np.array([0, 0, 1])
        cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
        r = cylinder.dimensions.get("radius", 0)
        
        # Get face properties
        face_normal = face.normal / np.linalg.norm(face.normal)
        face_pos = face.position
        bbox_min, bbox_max = face.bounding_box
        
        # Find where cylinder axis intersects the face plane
        # Line: P = cyl_pos + t * cyl_axis
        # Plane: dot(P - face_pos, face_normal) = 0
        
        denom = np.dot(cyl_axis, face_normal)
        if abs(denom) < 1e-6:  # Parallel to plane
            # Check if cylinder is in the plane
            dist = abs(np.dot(cyl_pos - face_pos, face_normal))
            if dist <= r + self.tolerance:
                # Check if cylinder position is near face bounds
                return self._point_near_face_bounds(cyl_pos, face)
            return False
        
        # Calculate intersection parameter
        t = np.dot(face_pos - cyl_pos, face_normal) / denom
        
        # Get intersection point
        intersection = cyl_pos + t * cyl_axis
        
        # Check if intersection is within face bounds (with margin for cylinder radius)
        margin = r + self.tolerance * 2
        for i in range(3):
            if intersection[i] < bbox_min[i] - margin or intersection[i] > bbox_max[i] + margin:
                return False
        
        return True
