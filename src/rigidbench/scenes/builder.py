from __future__ import annotations

import math
from functools import wraps
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

from .schema.assets import BLENDERKIT_DATA
from .schema.registry import get_registry

PASSIVE_GEOMETRY_PHYSICS = {
    "mass": 1.0,
    "friction": 0.4,
    "restitution": 0.3,
    "collision_shape": "MESH",
    "collision_margin": 0.0001,
}


class SceneBuilder:
    """Mutates a Blender scene as actors are placed for a task."""

    def __init__(self, surface, defer_physics: bool = False):
        self.surface = surface
        self.objects: list[bpy.types.Object] = []
        self.defer_physics = defer_physics
        self._bbox_cache: dict[str, float] = {}
        self._deferred: dict[str, dict] = {}

    def spawn(
        self,
        object_id,
        name,
        at=None,
        on=None,
        above=None,
        beside=None,
        active=True,
        align_to_surface=False,
    ):
        """Load object_id, place it (on / above / beside), and queue or attach physics."""
        obj = self._load_object(object_id, name)

        if beside:
            ref_obj, gap, surface = beside
            self._place_beside(obj, ref_obj, gap, surface)
        elif on:
            self._place_on(obj, at, on, align_to_surface=align_to_surface)
        elif above:
            surface, height = above
            z = self._raycast_z(at, surface) + height
            obj.location = (*at, z)

        self._register_physics(obj, object_id=object_id, active=active)
        self.objects.append(obj)
        return obj

    def create_ramp(self, name, at, on, length, height, width, material=None, facing="forward"):
        """Build a static ramp mesh, place it on a surface, attach passive physics."""
        obj = _make_straight_ramp(name, length, height, width)
        self._place_on(obj, at, on)
        if material:
            self._apply_material(obj, material)
        if facing == "backward":
            obj.rotation_euler.z = math.pi
            bpy.context.view_layer.update()
            self._place_on(obj, at, on)
        self._register_physics(obj, object_id=None, active=False, physics=PASSIVE_GEOMETRY_PHYSICS)
        self.objects.append(obj)
        return obj

    def bbox_radius(self, obj) -> float:
        """Half the larger of an object's XY bounding-box extents."""
        bpy.context.view_layer.update()
        coords = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        return (
            max(
                max(c.x for c in coords) - min(c.x for c in coords),
                max(c.y for c in coords) - min(c.y for c in coords),
            )
            / 2
        )

    def cached_bbox_radius(self, object_id: str) -> float:
        """bbox_radius for an asset without placing it, by spawning a throwaway probe."""
        if object_id not in self._bbox_cache:
            probe = self._load_object(object_id, "_bbox_probe")
            bpy.context.view_layer.update()
            self._bbox_cache[object_id] = self.bbox_radius(probe)
            bpy.data.objects.remove(probe, do_unlink=True)
        return self._bbox_cache[object_id]

    def apply_deferred_physics(self) -> None:
        """Drain the deferred queue, attaching a rigid body to each pending actor."""
        for obj in self.objects:
            spec = self._deferred.pop(obj.name, None)
            if spec is None:
                continue
            attach_rigid_body(
                obj,
                object_id=spec["object_id"],
                active=spec["active"],
                physics=spec.get("physics"),
            )

    def _load_object(self, object_id, name):
        registry = get_registry()
        spec = registry.object(object_id)
        scale = spec.asset.scale

        with bpy.data.libraries.load(str(registry.object_path(object_id))) as (data_from, data_to):
            data_to.objects = data_from.objects

        obj = next((o for o in data_to.objects if o and o.type == "MESH"), None)
        for o in data_to.objects:
            if o and o.type == "EMPTY":
                bpy.data.objects.remove(o, do_unlink=True)

        bpy.context.collection.objects.link(obj)
        obj.name = name
        obj["object_id"] = object_id

        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

        if scale != 1.0:
            obj.scale = (scale,) * 3
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(scale=True)

        _relink_textures(obj)
        bpy.context.view_layer.update()
        return obj

    def _apply_material(self, obj, material_id):
        mat_path = get_registry().material_path(material_id)
        with bpy.data.libraries.load(str(mat_path)) as (data_from, data_to):
            data_to.materials = data_from.materials

        mat = next((m for m in data_to.materials if m), None)
        if not mat:
            return

        if not obj.data.uv_layers:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
            bpy.ops.object.mode_set(mode="OBJECT")

        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        _relink_textures(obj)

    def _place_on(self, obj, at, target, align_to_surface=False):
        surface_z, normal = self._raycast_z_with_normal(at, target)
        bpy.context.view_layer.update()

        is_sloped = normal and normal.z < 0.99

        if align_to_surface and is_sloped:
            slope_angle_x = math.atan2(normal.y, normal.z)
            slope_angle_y = math.atan2(-normal.x, normal.z)
            obj.rotation_euler = (slope_angle_x, slope_angle_y, obj.rotation_euler.z)
            bpy.context.view_layer.update()

        bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        lowest_z = min(v.z for v in bbox_world)
        lowest_point = min(bbox_world, key=lambda v: v.z)

        if is_sloped:
            offset_x = lowest_point.x - obj.location.x
            offset_y = lowest_point.y - obj.location.y
            contact_xy = (at[0] + offset_x, at[1] + offset_y)
            contact_z, _ = self._raycast_z_with_normal(contact_xy, target)
        else:
            contact_z = surface_z

        origin_to_bottom_z = obj.location.z - lowest_z
        obj.location = (at[0], at[1], contact_z + origin_to_bottom_z + 0.002)
        bpy.context.view_layer.update()

    def _place_beside(self, obj, ref_obj, gap, surface):
        bpy.context.view_layer.update()
        ref_front_y = max((ref_obj.matrix_world @ v.co).y for v in ref_obj.data.vertices)

        obj.location = (ref_obj.location.x, ref_front_y + gap, ref_obj.location.z)
        bpy.context.view_layer.update()

        obj_back_y = min((obj.matrix_world @ v.co).y for v in obj.data.vertices)
        origin_to_back = obj.location.y - obj_back_y
        target_y = ref_front_y + gap + origin_to_back

        at = (ref_obj.location.x, target_y)
        surface_z, _ = self._raycast_z_with_normal(at, surface)
        min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        origin_to_bottom = obj.location.z - min_z
        obj.location = (at[0], at[1], surface_z + origin_to_bottom)
        bpy.context.view_layer.update()

    def _raycast_z_with_normal(self, at, target):
        """Drop a ray straight down at (at.x, at.y) onto target, returning (hit_z, world_normal)."""
        bpy.context.view_layer.update()
        origin_world = Vector((at[0], at[1], 1000))
        direction_world = Vector((0, 0, -1))

        origin_local = target.matrix_world.inverted() @ origin_world
        direction_local = (target.matrix_world.inverted().to_3x3() @ direction_world).normalized()

        success, loc, normal, _ = target.ray_cast(origin_local, direction_local)
        if success:
            hit_world = target.matrix_world @ loc
            normal_world = (target.matrix_world.to_3x3() @ normal).normalized()
            return hit_world.z, normal_world
        return max((target.matrix_world @ Vector(v)).z for v in target.bound_box), None

    def _raycast_z(self, at, target):
        z, _ = self._raycast_z_with_normal(at, target)
        return z

    def _register_physics(self, obj, object_id, active, physics=None):
        """Queue physics for deferred application, or attach immediately."""
        if self.defer_physics:
            spec = {"object_id": object_id, "active": active}
            if physics is not None:
                spec["physics"] = dict(physics)
            self._deferred[obj.name] = spec
        else:
            attach_rigid_body(obj, object_id, active, physics)


def _make_straight_ramp(name: str, length: float, height: float, width: float) -> bpy.types.Object:
    """Build a five-face wedge mesh: right-triangular cross-section along Y."""
    hw = width / 2
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    v = [
        bm.verts.new(p)
        for p in [
            (-hw, 0, 0),
            (hw, 0, 0),
            (-hw, 0, height),
            (hw, 0, height),
            (-hw, length, 0),
            (hw, length, 0),
        ]
    ]
    bm.faces.new([v[0], v[4], v[5], v[1]])
    bm.faces.new([v[0], v[1], v[3], v[2]])
    bm.faces.new([v[0], v[2], v[4]])
    bm.faces.new([v[1], v[5], v[3]])
    bm.faces.new([v[2], v[3], v[5], v[4]])

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj["object_id"] = "ramp"
    return obj


def _relink_textures(obj) -> None:
    """Repoint missing textures to the first matching filename under BLENDERKIT_DATA."""
    if not obj.data or not hasattr(obj.data, "materials"):
        return
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            img = node.image
            path = bpy.path.abspath(img.filepath) if img.filepath else ""
            if img.filepath and not Path(path).exists():
                filename = Path(img.filepath.replace("\\", "/")).name
                for found in BLENDERKIT_DATA.rglob(filename):
                    img.filepath = str(found)
                    img.reload()
                    break


def view3d_context() -> dict:
    """Find an active VIEW_3D area (some bpy.ops calls require this in their context)."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        return {"window": win, "area": area, "region": region}
    return {}


def _with_view3d(fn):
    """Decorator: run fn with a temp VIEW_3D context override."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with bpy.context.temp_override(**view3d_context()):
            return fn(*args, **kwargs)

    return wrapper


@_with_view3d
def attach_rigid_body(obj, object_id=None, active=True, physics=None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()

    if object_id is not None:
        p = get_registry().object(object_id).physics
        mass, friction, restitution = p.mass, p.friction, p.restitution
        collision_shape = p.collision_shape if active else "MESH"
    elif physics is not None:
        mass = physics["mass"]
        friction = physics["friction"]
        restitution = physics["restitution"]
        collision_shape = physics["collision_shape"]
    else:
        raise ValueError(f"physics undefined for {obj.name!r} (no object_id and no physics override)")

    margin = physics.get("collision_margin", 0.001) if physics is not None else 0.001

    rb = obj.rigid_body
    rb.type = "ACTIVE" if active else "PASSIVE"
    rb.mass = mass
    rb.friction = friction
    rb.restitution = restitution
    rb.collision_shape = collision_shape
    rb.collision_margin = margin
    rb.linear_damping = 0.1
    rb.angular_damping = 0.5
    # eval filters by obj["role"] == "active"
    obj["role"] = "active" if active else "passive"
