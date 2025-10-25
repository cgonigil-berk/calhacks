"""
3D Visualizer Module
Visualizes STEP assemblies with highlighted contact surfaces
"""

import numpy as np
from typing import List, Optional
from feature_analyzer import FeatureAnalyzer, FeatureType
from contact_detector import Contact
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class AssemblyVisualizer:
    """Visualizes STEP assemblies in 3D with contact highlighting"""
    
    def __init__(self, feature_analyzer: FeatureAnalyzer, contacts: List[Contact]):
        """
        Initialize visualizer
        
        Args:
            feature_analyzer: FeatureAnalyzer instance with loaded parts
            contacts: List of detected contacts
        """
        self.feature_analyzer = feature_analyzer
        self.contacts = contacts
        self.contact_feature_ids = set()
        
        # Extract contact feature IDs
        for contact in contacts:
            self.contact_feature_ids.add(contact.feature1.id)
            self.contact_feature_ids.add(contact.feature2.id)
    
    def visualize(self, output_file: Optional[str] = None, show: bool = True):
        """
        Create 3D visualization of the assembly
        
        Args:
            output_file: Optional path to save the figure
            show: Whether to display the figure
        """
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRep import BRep_Tool
        import cadquery as cq
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color scheme
        part_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 
                      'plum', 'khaki', 'lightgray', 'peachpuff']
        contact_color = 'red'
        
        print("Rendering 3D visualization...")
        
        # Get all features for each part
        features_by_part = {}
        for feature in self.feature_analyzer.features:
            if feature.part_id not in features_by_part:
                features_by_part[feature.part_id] = []
            features_by_part[feature.part_id].append(feature)
        
        # Counter for contact faces actually rendered
        contact_faces_rendered = 0
        
        # Render each part
        for part_idx, part in enumerate(self.feature_analyzer.parts):
            try:
                # Get the shape
                if hasattr(part, 'val'):
                    shape_obj = part.val()
                    if hasattr(shape_obj, 'wrapped'):
                        shape = shape_obj.wrapped
                    else:
                        shape = shape_obj
                else:
                    shape = part
                
                # Mesh the shape for visualization
                mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.1, True)
                mesh.Perform()
                
                # Explore faces
                face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
                face_count = 0
                
                while face_explorer.More():
                    occ_face = face_explorer.Current()
                    
                    # Check if this face is in contact
                    cq_face = cq.Face(occ_face)
                    is_contact = self._is_face_in_contact(cq_face, part_idx)
                    
                    # Get triangulation - need to convert TopoDS_Face
                    try:
                        from OCP.TopoDS import TopoDS
                        topo_face = TopoDS.Face_s(occ_face)
                        
                        location = TopLoc_Location()
                        triangulation = BRep_Tool.Triangulation_s(topo_face, location)
                        
                        if triangulation is None:
                            # Face not triangulated, skip
                            face_explorer.Next()
                            continue
                        
                        # Get transformation from location
                        trsf = location.Transformation()
                        
                        # Extract vertices
                        vertices = []
                        for i in range(1, triangulation.NbNodes() + 1):
                            pnt = triangulation.Node(i)
                            # Apply transformation
                            pnt_transformed = pnt.Transformed(trsf)
                            vertices.append([pnt_transformed.X(), pnt_transformed.Y(), pnt_transformed.Z()])
                        vertices = np.array(vertices)
                        
                        # Extract triangles
                        triangles = []
                        for i in range(1, triangulation.NbTriangles() + 1):
                            triangle = triangulation.Triangle(i)
                            idx = [triangle.Value(j) - 1 for j in range(1, 4)]
                            triangles.append(idx)
                        
                        if len(triangles) == 0:
                            face_explorer.Next()
                            continue
                        
                        # Create triangle mesh
                        tri_vertices = vertices[triangles]
                        
                        # Choose color
                        if is_contact:
                            color = contact_color
                            alpha = 0.9
                            edgecolor = 'darkred'
                            linewidth = 1.5
                            contact_faces_rendered += 1
                        else:
                            color = part_colors[part_idx % len(part_colors)]
                            alpha = 0.6
                            edgecolor = 'black'
                            linewidth = 0.1
                        
                        # Add to plot
                        collection = Poly3DCollection(tri_vertices, 
                                                     facecolors=color,
                                                     alpha=alpha,
                                                     edgecolors=edgecolor,
                                                     linewidths=linewidth)
                        ax.add_collection3d(collection)
                        
                        face_count += 1
                    
                    except Exception as e:
                        pass  # Skip faces that can't be triangulated
                    
                    face_explorer.Next()
                
                print(f"Part {part_idx}: rendered {face_count} faces")
                
            except Exception as e:
                print(f"Error rendering part {part_idx}: {e}")
        
        # Add feature markers for contact features
        self._add_contact_markers(ax)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('STEP Assembly with Contact Surfaces Highlighted\n(Red = Contact Surfaces)', 
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        self._set_axes_equal(ax)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.9, label=f'Contact Surfaces ({contact_faces_rendered} faces)'),
            Patch(facecolor='lightblue', alpha=0.6, label='Non-contact Surfaces')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_file}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
    
    def _is_face_in_contact(self, face, part_id: int) -> bool:
        """Check if a face is part of a contact - must match closely"""
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_SurfaceType
            
            center = face.Center()
            face_pos = np.array([center.x, center.y, center.z])
            
            # Get face surface type
            adaptor = BRepAdaptor_Surface(face.wrapped)
            face_surface_type = adaptor.GetType()
            
            # Check against all contacts
            for contact in self.contacts:
                # For clearance/interference fits (shaft in hole), both features are cylindrical
                if contact.contact_type in ["Clearance/Interference Fit", "Interference/Transition Fit"]:
                    # Check both features in the contact
                    for feature in [contact.feature1, contact.feature2]:
                        if feature.part_id != part_id:
                            continue
                        
                        # Feature must be a cylinder
                        if feature.feature_type not in [FeatureType.CYLINDER, FeatureType.HOLE]:
                            continue
                        
                        # This face must also be cylindrical
                        if face_surface_type != GeomAbs_SurfaceType.GeomAbs_Cylinder:
                            continue
                        
                        # Check if face radius matches feature radius
                        try:
                            face_cylinder = adaptor.Cylinder()
                            face_radius = face_cylinder.Radius()
                            feature_radius = feature.dimensions.get("radius", 0)
                            
                            # Radii must match closely (within 1%)
                            if abs(face_radius - feature_radius) < feature_radius * 0.01:
                                # Check if face center is close to feature position
                                distance = np.linalg.norm(face_pos - feature.position)
                                
                                # For cylindrical surfaces, centers should be on the axis
                                # So perpendicular distance to axis should be approximately the radius
                                if distance < feature_radius * 2:
                                    return True
                        except:
                            pass
                
                # For face-to-face contacts
                elif contact.contact_type in ["Mating Surfaces", "Surface Contact", "Perpendicular Contact"]:
                    for feature in [contact.feature1, contact.feature2]:
                        if feature.part_id != part_id:
                            continue
                        
                        if feature.feature_type != FeatureType.FACE:
                            continue
                        
                        # Check if face position matches feature position
                        distance = np.linalg.norm(face_pos - feature.position)
                        
                        if distance < 5.0:  # Within 5mm
                            # Check bounding box overlap
                            bbox = face.BoundingBox()
                            feat_bbox_min, feat_bbox_max = feature.bounding_box
                            
                            overlap_x = not (bbox.xmax < feat_bbox_min[0] or bbox.xmin > feat_bbox_max[0])
                            overlap_y = not (bbox.ymax < feat_bbox_min[1] or bbox.ymin > feat_bbox_max[1])
                            overlap_z = not (bbox.zmax < feat_bbox_min[2] or bbox.zmin > feat_bbox_max[2])
                            
                            if overlap_x and overlap_y and overlap_z:
                                return True
                
                # For edge contacts (cylinder-face)
                elif contact.contact_type == "Edge Contact":
                    for feature in [contact.feature1, contact.feature2]:
                        if feature.part_id != part_id:
                            continue
                        
                        # Check if this face matches the feature
                        distance = np.linalg.norm(face_pos - feature.position)
                        
                        if distance < 10.0:
                            bbox = face.BoundingBox()
                            feat_bbox_min, feat_bbox_max = feature.bounding_box
                            
                            overlap_x = not (bbox.xmax < feat_bbox_min[0] or bbox.xmin > feat_bbox_max[0])
                            overlap_y = not (bbox.ymax < feat_bbox_min[1] or bbox.ymin > feat_bbox_max[1])
                            overlap_z = not (bbox.zmax < feat_bbox_min[2] or bbox.zmin > feat_bbox_max[2])
                            
                            if overlap_x and overlap_y and overlap_z:
                                return True
            
        except Exception as e:
            pass
        
        return False
    
    def _add_contact_markers(self, ax):
        """Add markers showing contact feature centers"""
        for contact in self.contacts:
            # Draw line between contact features
            pos1 = contact.feature1.position
            pos2 = contact.feature2.position
            
            ax.plot([pos1[0], pos2[0]], 
                   [pos1[1], pos2[1]], 
                   [pos1[2], pos2[2]], 
                   'r-', linewidth=2, alpha=0.7)
            
            # Add markers at contact centers
            mid_point = (pos1 + pos2) / 2
            ax.scatter(*mid_point, c='red', s=100, marker='*', 
                      edgecolors='darkred', linewidth=1.5, zorder=10)
    
    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D plot"""
        # Get the current limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        # Calculate ranges
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        # Find the largest range
        max_range = max([x_range, y_range, z_range])
        
        # Calculate centers
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        
        # Set new limits
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])
    
    def create_contact_report_visualization(self, output_file: Optional[str] = None):
        """Create a visualization showing contact details"""
        if not self.contacts:
            print("No contacts to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Contact Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Contact types pie chart
        contact_types = {}
        for contact in self.contacts:
            ct = contact.contact_type
            contact_types[ct] = contact_types.get(ct, 0) + 1
        
        axes[0, 0].pie(contact_types.values(), labels=contact_types.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Contact Types Distribution')
        
        # 2. Contact distances histogram
        distances = [c.distance for c in self.contacts]
        axes[0, 1].hist(distances, bins=20, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Distance (mm)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Contact Distance Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Contact areas bar chart
        contact_labels = [f"C{i+1}" for i in range(len(self.contacts))]
        contact_areas = [c.contact_area for c in self.contacts]
        axes[1, 0].bar(contact_labels, contact_areas, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Contact ID')
        axes[1, 0].set_ylabel('Estimated Area (mm²)')
        axes[1, 0].set_title('Contact Areas')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Part interaction matrix
        n_parts = self.feature_analyzer.get_part_count()
        interaction_matrix = np.zeros((n_parts, n_parts))
        
        for contact in self.contacts:
            p1 = contact.feature1.part_id
            p2 = contact.feature2.part_id
            interaction_matrix[p1, p2] += 1
            interaction_matrix[p2, p1] += 1
        
        im = axes[1, 1].imshow(interaction_matrix, cmap='YlOrRd', interpolation='nearest')
        axes[1, 1].set_xlabel('Part ID')
        axes[1, 1].set_ylabel('Part ID')
        axes[1, 1].set_title('Part Interaction Matrix')
        axes[1, 1].set_xticks(range(n_parts))
        axes[1, 1].set_yticks(range(n_parts))
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], label='Number of Contacts')
        
        # Add text annotations
        for i in range(n_parts):
            for j in range(n_parts):
                if interaction_matrix[i, j] > 0:
                    axes[1, 1].text(j, i, int(interaction_matrix[i, j]),
                                  ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Contact report saved to: {output_file}")
        
        plt.show()
        
        return fig, axes


def visualize_assembly(step_file: str, tolerance: float = 0.01, 
                      output_3d: Optional[str] = None,
                      output_report: Optional[str] = None,
                      show: bool = True):
    """
    Convenience function to visualize a STEP assembly
    
    Args:
        step_file: Path to STEP file
        tolerance: Contact detection tolerance
        output_3d: Optional path to save 3D visualization
        output_report: Optional path to save contact report
        show: Whether to display the visualizations
    """
    from contact_detector import ContactDetector
    
    # Load and analyze
    print(f"Loading {step_file}...")
    analyzer = FeatureAnalyzer(step_file)
    
    print("Identifying features...")
    features = analyzer.identify_features()
    
    print("Detecting contacts...")
    detector = ContactDetector(features, tolerance=tolerance)
    contacts = detector.detect_contacts()
    
    print(f"Found {len(contacts)} contacts")
    
    # Create visualizations
    visualizer = AssemblyVisualizer(analyzer, contacts)
    
    # 3D assembly view
    visualizer.visualize(output_file=output_3d, show=show)
    
    # Contact report
    if contacts:
        visualizer.create_contact_report_visualization(output_file=output_report)
    else:
        print("No contacts detected - skipping contact report")
    
    return visualizer
