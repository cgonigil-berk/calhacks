"""
Tolerance Evaluator Module
Determines which contacts require tolerancing
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from contact_detector import Contact
from feature_analyzer import FeatureType


@dataclass
class ToleranceRequirement:
    """Represents a tolerancing requirement"""
    contact: Contact
    tolerance_type: str
    reason: str
    recommended_tolerance: str
    priority: str  # "Critical", "Important", "Standard"
    
    def __repr__(self):
        return f"ToleranceReq({self.tolerance_type}, {self.priority})"


class ToleranceEvaluator:
    """Evaluates contacts to determine tolerancing requirements"""
    
    # Tolerance rules based on contact types
    TOLERANCE_RULES = {
        "Clearance/Interference Fit": {
            "type": "Hole-Shaft Fit",
            "requires_tolerance": True,
            "priority": "Critical",
            "reason": "Cylindrical fit requires precise tolerance for proper assembly and function",
            "recommendation": "ISO fit designation (e.g., H7/h6 for precision, H8/f7 for clearance)"
        },
        "Interference/Transition Fit": {
            "type": "Hole-Shaft Fit",
            "requires_tolerance": True,
            "priority": "Critical",
            "reason": "Interference fit requires precise tolerance for proper press fit assembly",
            "recommendation": "ISO interference fit designation (e.g., H7/p6 for press fit, H7/s6 for force fit)"
        },
        "Mating Surfaces": {
            "type": "Flatness/Surface Profile",
            "requires_tolerance": True,
            "priority": "Important",
            "reason": "Mating surfaces require flatness and surface profile control",
            "recommendation": "Flatness tolerance ±0.05mm, Surface profile ±0.1mm"
        },
        "Edge Contact": {
            "type": "Positional Tolerance",
            "requires_tolerance": True,
            "priority": "Standard",
            "reason": "Edge contact requires position control to ensure proper alignment",
            "recommendation": "Position tolerance ±0.2mm"
        },
        "Line Contact": {
            "type": "Parallelism/Perpendicularity",
            "requires_tolerance": True,
            "priority": "Important",
            "reason": "Line contact requires geometric tolerance for proper alignment",
            "recommendation": "Parallelism or Perpendicularity ±0.1mm"
        },
        "Surface Contact": {
            "type": "Surface Profile",
            "requires_tolerance": True,
            "priority": "Standard",
            "reason": "Surface contact requires profile control",
            "recommendation": "Profile tolerance ±0.15mm"
        },
        "Perpendicular Contact": {
            "type": "Perpendicularity",
            "requires_tolerance": True,
            "priority": "Important",
            "reason": "Perpendicular surfaces require geometric control",
            "recommendation": "Perpendicularity ±0.1mm"
        },
        "General Contact": {
            "type": "General Dimensional",
            "requires_tolerance": True,
            "priority": "Standard",
            "reason": "Contact interface requires dimensional control",
            "recommendation": "General tolerance ±0.2mm"
        }
    }
    
    def __init__(self, contacts: List[Contact]):
        """
        Initialize tolerance evaluator
        
        Args:
            contacts: List of contacts to evaluate
        """
        self.contacts = contacts
        self.tolerance_requirements = []
    
    def evaluate_tolerances(self) -> List[ToleranceRequirement]:
        """Evaluate all contacts and determine tolerancing requirements"""
        self.tolerance_requirements = []
        
        for contact in self.contacts:
            requirement = self._evaluate_contact(contact)
            if requirement:
                self.tolerance_requirements.append(requirement)
        
        return self.tolerance_requirements
    
    def _evaluate_contact(self, contact: Contact) -> ToleranceRequirement:
        """Evaluate a single contact for tolerancing requirements"""
        
        contact_type = contact.contact_type
        
        # Get tolerance rule for this contact type
        if contact_type in self.TOLERANCE_RULES:
            rule = self.TOLERANCE_RULES[contact_type]
            
            if rule["requires_tolerance"]:
                # Apply additional rules based on specific feature combinations
                tolerance_type = rule["type"]
                reason = rule["reason"]
                recommendation = rule["recommendation"]
                priority = rule["priority"]
                
                # Adjust for critical fits
                if contact_type in ["Clearance/Interference Fit", "Interference/Transition Fit"]:
                    recommendation = self._get_fit_recommendation(contact)
                    priority = "Critical"
                
                # Adjust priority based on contact characteristics
                if contact.contact_area > 100:  # Large contact area
                    priority = self._elevate_priority(priority)
                
                return ToleranceRequirement(
                    contact=contact,
                    tolerance_type=tolerance_type,
                    reason=reason,
                    recommended_tolerance=recommendation,
                    priority=priority
                )
        
        return None
    
    def _get_fit_recommendation(self, contact: Contact) -> str:
        """Get specific fit recommendation for hole-shaft contact"""
        
        # Extract dimensions from features
        feature1 = contact.feature1
        feature2 = contact.feature2
        
        r1 = feature1.dimensions.get("radius", 0)
        r2 = feature2.dimensions.get("radius", 0)
        clearance = abs(r1 - r2)
        diameter = (r1 + r2)  # Average diameter
        
        # Determine fit type based on clearance
        if clearance < 0.01:  # Interference or press fit (< 0.01mm clearance)
            if diameter < 10:
                return "H7/p6 (Press fit)"
            elif diameter < 50:
                return "H7/s6 (Force fit)"
            else:
                return "H7/u6 (Heavy press fit)"
        
        elif clearance < 0.1:  # Transition fit (0.01 - 0.1mm clearance)
            if diameter < 10:
                return "H7/k6 (Transition fit)"
            elif diameter < 50:
                return "H7/n6 (Sliding fit)"
            else:
                return "H7/m6 (Light transition fit)"
        
        elif clearance < 0.5:  # Close running fit
            if diameter < 10:
                return "H7/g6 (Close running fit)"
            elif diameter < 50:
                return "H8/f7 (Close running fit)"
            else:
                return "H8/e8 (Close running fit)"
        
        else:  # Free running fit
            if diameter < 6:
                return "H7/h6 (Precision fit for small diameters)"
            elif diameter < 30:
                return "H7/g6 (Close running fit)"
            elif diameter < 100:
                return "H8/f7 (Sliding fit)"
            else:
                return "H9/e8 (Loose running fit for large diameters)"
    
    def _elevate_priority(self, current_priority: str) -> str:
        """Elevate priority level if needed"""
        if current_priority == "Standard":
            return "Important"
        elif current_priority == "Important":
            return "Critical"
        return current_priority
    
    def get_critical_requirements(self) -> List[ToleranceRequirement]:
        """Get only critical tolerance requirements"""
        return [req for req in self.tolerance_requirements 
                if req.priority == "Critical"]
    
    def get_requirements_by_priority(self) -> Dict[str, List[ToleranceRequirement]]:
        """Group requirements by priority"""
        by_priority = {
            "Critical": [],
            "Important": [],
            "Standard": []
        }
        
        for req in self.tolerance_requirements:
            by_priority[req.priority].append(req)
        
        return by_priority
    
    def get_requirements_by_type(self) -> Dict[str, List[ToleranceRequirement]]:
        """Group requirements by tolerance type"""
        by_type = {}
        
        for req in self.tolerance_requirements:
            if req.tolerance_type not in by_type:
                by_type[req.tolerance_type] = []
            by_type[req.tolerance_type].append(req)
        
        return by_type
    
    def get_summary_statistics(self) -> Dict[str, int]:
        """Get summary statistics of tolerance requirements"""
        stats = {
            "total_requirements": len(self.tolerance_requirements),
            "critical": len([r for r in self.tolerance_requirements if r.priority == "Critical"]),
            "important": len([r for r in self.tolerance_requirements if r.priority == "Important"]),
            "standard": len([r for r in self.tolerance_requirements if r.priority == "Standard"])
        }
        
        return stats
