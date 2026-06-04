import numpy as np
import trimesh

from granular_analysis.utils.mesh import section_area_from_mesh, extract_3d_slice_areas


def test_box_section_area():
    # Create a box of size 2 x 3 x 4 centered at origin
    box = trimesh.creation.box(extents=(2.0, 3.0, 4.0))

    # slice at z=0 should give rectangle area 2*3 = 6
    section = box.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    area = section_area_from_mesh(section)
    assert area > 0
    assert abs(area - 6.0) / 6.0 < 0.05


def test_extract_3d_slice_areas():
    box = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    z_levels = np.array([-1.0, 0.0, 1.0])
    areas = extract_3d_slice_areas(box, z_levels)
    assert len(areas) == 3
    assert abs(areas[1] - 6.0) / 6.0 < 0.05


if __name__ == '__main__':
    test_box_section_area()
    test_extract_3d_slice_areas()
    print('test_mesh_utils: OK')
