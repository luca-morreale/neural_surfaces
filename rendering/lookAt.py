
import bpy


def lookAt(camera, rotation=None, point=None):
	if rotation is None:
		direction = point - camera.location
		rotQuat = direction.to_track_quat('-Z', 'Y')
		camera.rotation_euler = rotQuat.to_euler()
	else:
		camera.rotation_euler = rotation