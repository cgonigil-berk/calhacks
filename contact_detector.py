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
        """Check if two features are in contact"""
        
        # Calculate distance between feature centers
        distance = np.linalg.norm(feature1.position - feature2.position)
        
        # Check bounding box overlap
        bbox1_min, bbox1_max = feature1.bounding_box
        bbox2_min, bbox2_max = feature2.bounding_box
        
        # Check if bounding boxes overlap or are very close
        overlap_x = self._check_overlap_1d(bbox1_min[0], bbox1_max[0], 
                                           bbox2_min[0], bbox2_max[0])
        overlap_y = self._check_overlap_1d(bbox1_min[1], bbox1_max[1], 
                                           bbox2_min[1], bbox2_max[1])
        overlap_z = self._check_overlap_1d(bbox1_min[2], bbox1_max[2], 
                                           bbox2_min[2], bbox2_max[2])
        
        if not (overlap_x and overlap_y and overlap_z):
            return None
        
        # Determine contact type based on feature types and geometry
        contact_type = self._determine_contact_type(feature1, feature2)
        
        # Estimate contact area
        contact_area = self._estimate_contact_area(feature1, feature2)
        
        # If distance is within tolerance and there's bbox overlap, consider it contact
        if distance < self._get_contact_threshold(feature1, feature2):
            return Contact(
                feature1=feature1,
                feature2=feature2,
                contact_type=contact_type,
                distance=distance,
                contact_area=contact_area
            )
        
        return None
    
    def _check_overlap_1d(self, min1: float, max1: float, 
                          min2: float, max2: float) -> bool:
        """Check if two 1D ranges overlap with tolerance"""
        return not (max1 + self.tolerance < min2 or max2 + self.tolerance < min1)
    
    def _determine_contact_type(self, feature1: Feature, feature2: Feature) -> str:
        """Determine the type of contact between features"""
        
        type1 = feature1.feature_type
        type2 = feature2.feature_type
        
        # Hole-to-cylinder contact (shaft in hole)
        if (type1 == FeatureType.HOLE and type2 == FeatureType.CYLINDER) or \
           (type1 == FeatureType.CYLINDER and type2 == FeatureType.HOLE):
            return "Clearance/Interference Fit"
        
        # Face-to-face contact
        if type1 == FeatureType.FACE and type2 == FeatureType.FACE:
            # Check if normals are opposite (facing contact)
            if feature1.normal is not None and feature2.normal is not None:
                dot_product = np.dot(feature1.normal, feature2.normal)
                if dot_product < -0.8:  # Nearly opposite
                    return "Mating Surfaces"
                elif abs(dot_product) < 0.2:  # Perpendicular
                    return "Perpendicular Contact"
            return "Surface Contact"
        
        # Cylinder-to-face contact
        if (type1 == FeatureType.CYLINDER and type2 == FeatureType.FACE) or \
           (type1 == FeatureType.FACE and type2 == FeatureType.CYLINDER):
            return "Edge Contact"
        
        # Cylinder-to-cylinder contact
        if type1 == FeatureType.CYLINDER and type2 == FeatureType.CYLINDER:
            return "Line Contact"
        
        return "General Contact"
    
    def _get_contact_threshold(self, feature1: Feature, feature2: Feature) -> float:
        """Get distance threshold for contact based on feature types"""
        
        # For cylindrical features, use larger threshold based on dimensions
        if feature1.feature_type == FeatureType.CYLINDER and \
           feature2.feature_type == FeatureType.CYLINDER:
            r1 = feature1.dimensions.get("radius", 0)
            r2 = feature2.dimensions.get("radius", 0)
            return r1 + r2 + self.tolerance
        
        if feature1.feature_type == FeatureType.HOLE and \
           feature2.feature_type == FeatureType.CYLINDER:
            r1 = feature1.dimensions.get("radius", 0)
            r2 = feature2.dimensions.get("radius", 0)
            return abs(r1 - r2) + self.tolerance
        
        # For face contacts, use smaller threshold
        if feature1.feature_type == FeatureType.FACE or \
           feature2.feature_type == FeatureType.FACE:
            return self.tolerance * 5
        
        return self.tolerance * 10
    
    def _estimate_contact_area(self, feature1: Feature, feature2: Feature) -> float:
        """Estimate the contact area between two features"""
        
        # For face-to-face contact, estimate overlap area
        if feature1.feature_type == FeatureType.FACE and \
           feature2.feature_type == FeatureType.FACE:
            area1 = feature1.dimensions.get("area", 0)
            area2 = feature2.dimensions.get("area", 0)
            return min(area1, area2) * 0.8  # Assume 80% overlap
        
        # For hole-cylinder contact, use circumference
        if (feature1.feature_type == FeatureType.HOLE and 
            feature2.feature_type == FeatureType.CYLINDER) or \
           (feature1.feature_type == FeatureType.CYLINDER and 
            feature2.feature_type == FeatureType.HOLE):
            r = min(feature1.dimensions.get("radius", 0),
                   feature2.dimensions.get("radius", 0))
            length = 10  # Assumed contact length in mm
            return 2 * np.pi * r * length
        
        return 0.0
    
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
