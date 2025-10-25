#!/usr/bin/env python3
"""
Generate a sample STEP assembly for testing
Creates a simple assembly with a base plate and a cylindrical shaft
"""

import cadquery as cq

# Create base plate with holes
base = (cq.Workplane("XY")
        .box(100, 100, 10)
        .faces(">Z").workplane()
        .pushPoints([(-30, 0), (30, 0)])
        .hole(10))

# Create cylindrical shaft
shaft = (cq.Workplane("XY")
         .workplane(offset=10)
         .circle(4.95)  # Slightly smaller than hole for clearance fit
         .extrude(20))

# Create top plate
top_plate = (cq.Workplane("XY")
            .workplane(offset=30)
            .box(80, 80, 5))

# Create assembly
assembly = cq.Assembly()
assembly.add(base, name="base", color=cq.Color("gray"))
assembly.add(shaft, name="shaft", loc=cq.Location((30, 0, 10)), color=cq.Color("blue"))
assembly.add(top_plate, name="top_plate", color=cq.Color("red"))

# Export as STEP
assembly.save("sample_assembly.step")
print("Sample assembly created: sample_assembly.step")

# Also export individual parts for testing
cq.exporters.export(base, "sample_base.step")
cq.exporters.export(shaft, "sample_shaft.step")
print("Individual parts created: sample_base.step, sample_shaft.step")
