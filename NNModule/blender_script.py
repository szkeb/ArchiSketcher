import bpy
from mathutils import Vector
from math import pi
import numpy as np
import os
import sys
from image_processing import copy_file, combine_images
from enum import Enum

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import config
from utils import create_dir_if_not_exists, delete_all_files_from


class Phase(Enum):
    One_Shape = 1
    Without_Clasters = 2
    One_Claster = 3
    Multiple_Clusters = 4


def append_claster_and_bool(primitive, claster=1, bool=1):
    primitive['claster'] = claster
    primitive['bool'] = bool
    return primitive

def generate_random_scenes(shape_limit, configuration, num_of_scenes, angle_steps, num_of_angles, min_angle_offset, phase):
    """
    Generates random scenes.
    """
    num_of_shapes = np.random.randint(1, shape_limit + 1, size=num_of_scenes)

    def random_angles(n_angles):
        angles = np.random.choice(np.arange(angle_steps), n_angles, replace=False)
        return list(angles * ((np.pi * 2) / angle_steps))

    if phase == Phase.One_Shape:
        return [
            ([append_claster_and_bool(generate_random_primitive(configuration)) for _ in range(1)], random_angles(num_of_angles))
            for _ in range(num_of_scenes)
        ]
    elif phase == Phase.Without_Clasters:
        return [
            ([append_claster_and_bool(generate_random_primitive(configuration)) for _ in range(num_of_shapes[i])], random_angles(num_of_angles))
            for i in range(num_of_scenes)
        ]
    elif phase == Phase.One_Claster:
        return [
            (generate_plus_minus_primitives(configuration), random_angles(num_of_angles))
            for i in range(num_of_scenes)
        ]
    elif phase == Phase.Multiple_Clusters:
        def random_clasters():
            pm1 = generate_plus_minus_primitives(configuration)
            pm2 = generate_plus_minus_primitives(configuration)
            pm1_dist_from_origo = pm1[0]['translation'][-1] ** 2 + pm1[0]['translation'][-2] ** 2
            pm2_dist_from_origo = pm2[0]['translation'][-1] ** 2 + pm2[0]['translation'][-2] ** 2
            cidx1 = 1 if pm1_dist_from_origo < pm2_dist_from_origo else 2
            cidx2 = 2 if cidx1 == 1 else 1
            pm1[0]['claster'] = cidx1
            pm1[1]['claster'] = cidx1
            pm2[0]['claster'] = cidx2
            pm2[1]['claster'] = cidx2
            return pm1 + pm2

        return [
            (random_clasters(), random_angles(num_of_angles))
            for _ in range(num_of_scenes)
        ]

    return


def generate_plus_minus_primitives(configuration):
    plus_primitive = generate_random_primitive(configuration, 0, 2)

    m_type = np.random.choice([0, 1])

    sx = np.random.uniform(1., 2.5)
    sy = np.random.uniform(1., 2.5)
    r = np.random.uniform(0., configuration['max_rotation'])
    minus_primitive = {'type': m_type,  # cube or cylinder
                       'scale': [sx, sy, configuration['max_scale']],
                       'rotation': r,
                       'translation': [0., 0.]}

    plus_center = plus_primitive['translation']
    # Inside
    magnitude_x = np.random.uniform(-1., 1.)
    magnitude_y = np.random.uniform(-1., 1.)
    point_inside = np.asarray([magnitude_x * plus_primitive['scale'][0], 0]) + np.asarray([0, magnitude_y * plus_primitive['scale'][1]])

    relative_point = point_inside

    r = plus_primitive['rotation']
    rotation_matrix = np.array([[np.cos(r), np.sin(r)],
                                [-np.sin(r), np.cos(r)]])
    relative_center = np.matmul(relative_point, rotation_matrix)

    center = np.asarray(plus_center) + relative_center
    minus_primitive['translation'] = [center[0], center[1]]

    plus_primitive['claster'] = 1
    minus_primitive['claster'] = 1
    plus_primitive['bool'] = 1
    minus_primitive['bool'] = 0

    return [plus_primitive, minus_primitive]


def generate_random_primitive(configuration, type=None, min_step=None):
    """
    Generates a random primitive (random type, scale, rotation, translation)
    according to the configuration set in the parameter.
    """
    max_scale = configuration['max_scale']
    max_rotation = configuration['max_rotation']
    max_translation = configuration['max_translation']

    scale_step = configuration['scale_step']
    rotation_step = configuration['rotation_step']
    translation_step = configuration['translation_step']

    primitive_type = np.random.randint(config.NUM_OF_PRIMITIVE_TYPES) if type is None else type
    primitive = {'type': primitive_type,
                 'scale': [1., 1., 1.],
                 'rotation': 0.,
                 'translation': [0., 0.]}

    min_scale = 1 if min_step is None else 1 + min_step * scale_step
    sx = np.random.choice(np.arange(min_scale, max_scale + scale_step, scale_step))
    sy = np.random.choice(np.arange(min_scale, max_scale + scale_step, scale_step))
    sz = np.random.choice(np.arange(min_scale, max_scale + scale_step, scale_step))
    primitive['scale'] = [sx, sy, sz]

    r = np.random.choice(np.arange(0, max_rotation, rotation_step))
    primitive['rotation'] = r

    tx = np.random.choice(np.arange(0, max_translation + translation_step, translation_step))
    ty = np.random.choice(np.arange(0, max_translation + translation_step, translation_step))
    primitive['translation'] = [tx, ty]

    return primitive


def write_scenes_to_file(fname, scenes):
    """
    Writes the given scene descriptors into a file.
    """
    print('Writing scenes to file ', fname)
    with open(fname, 'a') as file:
        for (primitives, camera_angles) in scenes:
            # flat scene: [1, p/m, type, sx, sy, sz, r, tx, ty] type=type+1 because 0 is reserved for nothing
            scene_flat = [[primitive['claster'], primitive['bool'], primitive['type'] + 1, *primitive['scale'], primitive['rotation'], *primitive['translation']] for primitive in primitives]

            # sort scenes by the distance from (0, 0) // tx ^ 2 + ty ^ 2
            scene_flat = sorted(scene_flat, key=lambda x: x[-1] * x[-1] + x[-2] * x[-2], reverse=True)
            # padding matrix with 'nothing' shapes
            if len(scene_flat) < config.NUM_OF_SHAPES:
                scene_flat += [[0 for _ in range(3 + config.NUM_OF_PARAMETERS)] for _ in range(config.NUM_OF_SHAPES - len(primitives))]

            # Constructing scene string
            scene_string = f'{str(camera_angles)};{str(scene_flat)}\n'
            file.write(scene_string)


def render_scenes(scenes, distance, output_dir, material, continue_from=0):
    print('Rendering scenes...')
    camera_setup(distance)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    padding = 6
    path = str(output_dir / "render_{0}-{1}.png")
    for i, (primitives, camera_angles) in enumerate(scenes):
        reset()
        meshes = [[] for _ in range(config.NUM_OF_SHAPES)]
        for primitive in primitives:
            create_primitive(primitive, meshes)

        assemble_scene(meshes)
        apply_material(material)

        # Rendering top and side view
        for ci, cam_name in enumerate(["Top", "Side"]):
            bpy.context.scene.camera = bpy.context.scene.objects.get(cam_name)
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(path.format(str(i+continue_from).zfill(padding), ci))

        # Rendering all the perspective views
        for ci, camera_angle in enumerate(camera_angles):
            move_perspective_camera(distance, camera_angle)
            bpy.context.scene.camera = bpy.context.scene.objects.get("Perspective")
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(path.format(str(i + continue_from).zfill(padding), ci + 2))


def camera_setup(distance):
    top = bpy.data.objects["Top"]
    side = bpy.data.objects["Side"]

    bpy.data.cameras["Top"].type = "ORTHO"
    bpy.data.cameras["Side"].type = "ORTHO"
    bpy.data.cameras["Perspective"].type = "PERSP"

    # Setting the locations of the cameras
    top.location = [0., 0., distance]
    side.location = [0., -distance, 0.]

    top.rotation_euler = [0., 0., 0.]
    side.rotation_euler = [pi / 2., 0., 0.]


def move_perspective_camera(distance, persp_angle):
    persp = bpy.data.objects["Perspective"]
    d = distance * 1.5
    persp.location = [d * np.cos(persp_angle), d * np.sin(persp_angle), d / 2]
    persp.rotation_euler = look_to([0., 0., 0.], persp.location)


def look_to(point, camera_pos):
    point = Vector(point)
    camera = Vector(camera_pos)

    difv = point - camera

    rot_quat = difv.to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()


def reset():
    for m in bpy.data.meshes:
        bpy.data.meshes.remove(m)


def create_primitive(primitive, meshes):
    location_xy = np.array(primitive['translation']) - config.MAX_TRANSLATION / 2.
    location_z = primitive['scale'][-1]

    shape = primitive['type']
    if shape == 0:
        bpy.ops.mesh.primitive_cube_add(
            scale=primitive['scale'],
            rotation=[0., 0., primitive['rotation']],
            location=[*location_xy, location_z])
    elif shape == 1:
        bpy.ops.mesh.primitive_cylinder_add(
            scale=primitive['scale'],
            rotation=[0., 0., primitive['rotation']],
            location=[*location_xy, location_z])
    elif shape == 2:
        bpy.ops.mesh.primitive_uv_sphere_add(
            scale=primitive['scale'],
            rotation=[0., 0., primitive['rotation']],
            location=[*location_xy, location_z])
    else:
        raise Exception(f"Primitive type {primitive['type']} does not exists.")

    cidx = primitive['claster']
    # If the claster index indicates an invalid row, we skip the primitive
    if cidx == 0:
        return
    bpy.context.active_object.name = f"{cidx}_{primitive['bool']}_{primitive['type']}_{len(meshes[cidx-1])}"
    meshes[cidx-1].append((primitive, bpy.context.active_object))
    return


def assemble_scene(meshes):
    # Phase 0
    # union_primitives()
    # Phase 1
    claster_subtract(meshes)
    return


def union_primitives():
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    first = meshes[0]
    bpy.context.view_layer.objects.active = first
    if len(meshes) > 1:
        for mesh in meshes[1:]:
            bm = first.modifiers.new('BM', 'BOOLEAN')
            bm.operation = 'UNION'
            bm.object = mesh
            bpy.ops.object.modifier_apply(modifier="BM")

            bpy.data.objects.remove(mesh)


def claster_subtract(meshes):
    for claster in [claster for claster in meshes if len(claster) != 0]:
        pluses = [obj for p, obj in claster if p['bool'] == 1]
        if len(pluses) == 0:
            continue
        active = pluses[0]
        bpy.context.view_layer.objects.active = active

        # Union all the pluses:
        for mesh in pluses[1:]:
            bm = active.modifiers.new('BM', 'BOOLEAN')
            bm.operation = 'UNION'
            bm.object = mesh
            bpy.ops.object.modifier_apply(modifier="BM")

            bpy.data.objects.remove(mesh)

        # Subtract all the minuses
        for mesh in [obj for p, obj in claster if p['bool'] == 0]:
            bm = active.modifiers.new('BM', 'BOOLEAN')
            bm.operation = 'DIFFERENCE'
            bm.object = mesh
            bpy.ops.object.modifier_apply(modifier="BM")

            bpy.data.objects.remove(mesh)


def apply_material(material_name):
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        material = bpy.data.materials.get(material_name)
        o.data.materials.append(material)


if __name__ == "__main__":
    ###############################################
    num_of_images_to_generate = config.DS_SIZE
    continue_from = 0  # use the number of previously rendered images (the first idx will be continue_from+0)
    combine = True
    phase = Phase.Multiple_Clusters
    ###############################################

    bpy.ops.wm.open_mainfile(filepath="sketch.blend")

    if continue_from == 0:
        # delete all files before generating
        delete_all_files_from(config.BLENDER_DIR)

    # generating programs
    generated_scenes = generate_random_scenes(
        shape_limit=config.NUM_OF_SHAPES,
        configuration={
            'max_scale': config.MAX_SCALE,
            'max_rotation': config.MAX_ROTATION,
            'max_translation': config.MAX_TRANSLATION,
            'scale_step': config.SCALE_STEP,
            'rotation_step': config.ROTATION_STEP,
            'translation_step': config.TRANSLATION_STEP
        },
        num_of_scenes=num_of_images_to_generate,
        angle_steps=config.ANGLE_STEPS,
        num_of_angles=config.NUM_OF_ANGLES,
        min_angle_offset=config.MIN_ANGLE_OFFSET,
        phase=phase
    )
    write_scenes_to_file(config.DS_PROGRAMS_BLENDER, generated_scenes)
    render_scenes(generated_scenes, config.DISTANCE, config.BLENDER_DIR, material='Transparent', continue_from=continue_from)

    print("---FINISHED RENDERING SCENES---")

    if combine:
        create_dir_if_not_exists(config.DS_DIR)
        delete_all_files_from(config.DS_IMAGES)
        copy_file(config.DS_PROGRAMS_BLENDER, config.DS_PROGRAMS)
        combine_images(config.BLENDER_DIR, config.DS_IMAGES)