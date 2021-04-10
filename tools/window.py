"""OpenGL Pointcloud viewer with http://pyglet.org.

Functions and classes are largely derived from
https://github.com/IntelRealSense/librealsense/blob/81d469db173dd682d3bada9bd7c7570db0f7cf76/wrappers/python/examples/pyglet_pointcloud_viewer.py

Usage of class Window:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.

Keyboard:
    [p]         Pause
    [r]         Reset View
    [z]         Toggle point scaling
    [x]         Toggle point distance attenuation
    [l]         Toggle lighting
    [1/2/3/...] Toggle camera switch
    [k]         Toggle point mask
    [m]         Toggle YCB/MANO mesh
    [s]         Save PNG (./out.png)
    [q/ESC]     Quit
"""

import numpy as np
import math
import pyglet
import pyglet.gl as gl
from transforms3d.quaternions import mat2quat, quat2mat


# https://stackoverflow.com/a/6802723
def rotation_matrix(axis, theta):
  """
  Returns the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  """
  axis = np.asarray(axis)
  axis = axis / math.sqrt(np.dot(axis, axis))
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class AppState:

  def __init__(self, *args, **kwargs):
    self.pitch, self.yaw = math.radians(-10), math.radians(-15)
    self.translation = np.array([0, 0, 1], np.float32)
    self.distance = 2
    self.mouse_btns = [False, False, False]
    self.paused = False
    self.scale = True
    self.attenuation = False
    self.lighting = False
    self.camera_off = [False] * kwargs['num_cameras']
    self.mask = 0
    self.model_off = False

  def reset(self):
    self.pitch, self.yaw, self.distance = 0, 0, 2
    self.translation[:] = 0, 0, 1

  @property
  def rotation(self):
    Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
    Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
    return np.dot(Ry, Rx).astype(np.float32)


def axes(size=1, width=1, pose=np.eye(4)):
  """Draws 3d axes."""

  # transform the vertices (6 x 4) by pose
  vertices = np.array([[0, 0, 0, 1], [size, 0, 0, 1],
                       [0, 0, 0, 1], [0, size, 0, 1],
                       [0, 0, 0, 1], [0, 0, size, 1]])
  vertices = np.matmul(pose, vertices.transpose()).transpose()
  v3f = vertices[:, :3].flatten()

  gl.glLineWidth(width)
  pyglet.graphics.draw(6, gl.GL_LINES,
                       ('v3f', v3f),
                       ('c3f', (1, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 1,
                                ))
                       )


def frustum(dimensions, intrinsics):
  """Draws camera's frustum."""
  w, h = dimensions[0], dimensions[1]
  batch = pyglet.graphics.Batch()

  for d in range(1, 6, 2):

    def get_point(x, y):
      p = list(np.linalg.inv(intrinsics).dot([x, y, 1]) * d)
      batch.add(2, gl.GL_LINES, None, ('v3f', [0, 0, 0] + p))
      return p

    top_left = get_point(0, 0)
    top_right = get_point(w, 0)
    bottom_right = get_point(w, h)
    bottom_left = get_point(0, h)

    batch.add(2, gl.GL_LINES, None, ('v3f', top_left + top_right))
    batch.add(2, gl.GL_LINES, None, ('v3f', top_right + bottom_right))
    batch.add(2, gl.GL_LINES, None, ('v3f', bottom_right + bottom_left))
    batch.add(2, gl.GL_LINES, None, ('v3f', bottom_left + top_left))

  batch.draw()


def grid(size=1, n=10, width=1):
  """Draws a grid on xz plane."""
  gl.glLineWidth(width)
  s = size / float(n)
  s2 = 0.5 * size
  batch = pyglet.graphics.Batch()

  for i in range(0, n + 1):
    x = -s2 + i * s
    batch.add(2, gl.GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
  for i in range(0, n + 1):
    z = -s2 + i * s
    batch.add(2, gl.GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

  batch.draw()


# TODO(ywchao): in future dev should be RGBPointCloudFromMultiCamerasWindow.
class Window():

  def __init__(self, dataloader):
    self.dataloader = dataloader

    self.config = gl.Config(double_buffer=True, samples=8)  # MSAA
    self.window = pyglet.window.Window(config=self.config, resizable=True)

    self.state = AppState(num_cameras=self.dataloader.num_cameras)

    self.pcd_vlist = []
    self.pcd_image = []
    self.pcd_meta = []
    w, h = self.dataloader.dimensions
    for _ in range(self.dataloader.num_cameras):
      self.pcd_vlist.append(
          pyglet.graphics.vertex_list(w * h, 'v3f/stream', 't2f/stream',
                                      'n3f/stream'))
      self.pcd_image.append(
          pyglet.image.ImageData(w, h, 'RGB', (gl.GLubyte * (w * h * 3))()))

      self.pcd_meta.append(None)

    self.fps_display = pyglet.clock.ClockDisplay()

    @self.window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
      w, h = map(float, self.window.get_size())

      if buttons & pyglet.window.mouse.LEFT:
        self.state.yaw -= dx * 0.5
        self.state.pitch -= dy * 0.5

      if buttons & pyglet.window.mouse.RIGHT:
        dp = np.array((dx / w, -dy / h, 0), np.float32)
        self.state.translation += np.dot(self.state.rotation, dp)

      if buttons & pyglet.window.mouse.MIDDLE:
        dz = dy * 0.01
        self.state.translation -= (0, 0, dz)
        self.state.distance -= dz

    @self.window.event
    def handle_mouse_btns(x, y, button, modifiers):
      self.state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
      self.state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
      self.state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)

    self.window.on_mouse_press = self.window.on_mouse_release = handle_mouse_btns

    @self.window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
      dz = scroll_y * 0.1
      self.state.translation -= (0, 0, dz)
      self.state.distance -= dz

    @self.window.event
    def on_key_press(symbol, modifiers):
      if symbol == pyglet.window.key.R:
        self.state.reset()

      if symbol == pyglet.window.key.P:
        self.state.paused ^= True

      if symbol == pyglet.window.key.Z:
        self.state.scale ^= True

      if symbol == pyglet.window.key.X:
        self.state.attenuation ^= True

      if symbol == pyglet.window.key.L:
        self.state.lighting ^= True

      for c in range(len(self.state.camera_off)):
        # _1, _2, _3, ...
        if symbol == 49 + c:
          self.state.camera_off[c] ^= True

      if symbol == pyglet.window.key.K:
        self.state.mask += 1
        if self.load_ycb and self.load_mano:
          self.state.mask %= 4
        elif self.load_ycb ^ self.load_mano:
          self.state.mask %= 3
        else:
          self.state.mask %= 2

      if symbol == pyglet.window.key.M and (self.load_ycb or self.load_mano):
        self.state.model_off ^= True

      if symbol == pyglet.window.key.S:
        pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')

      if symbol == pyglet.window.key.Q:
        self.window.close()

    @self.window.event
    def on_draw():
      self.window.clear()

      gl.glEnable(gl.GL_DEPTH_TEST)
      gl.glEnable(gl.GL_LINE_SMOOTH)

      width, height = self.window.get_size()
      gl.glViewport(0, 0, width, height)

      # Set projection matrix stack.
      gl.glMatrixMode(gl.GL_PROJECTION)
      gl.glLoadIdentity()
      gl.gluPerspective(60, width / float(height), 0.01, 20)

      # Set modelview matrix stack.
      gl.glMatrixMode(gl.GL_MODELVIEW)
      gl.glLoadIdentity()

      gl.gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

      gl.glTranslatef(0, 0, self.state.distance)
      gl.glRotated(self.state.pitch, 1, 0, 0)
      gl.glRotated(self.state.yaw, 0, 1, 0)

      if any(self.state.mouse_btns):
        axes(0.1, 4)

      gl.glTranslatef(0, 0, -self.state.distance)
      gl.glTranslatef(*self.state.translation)

      # Draw grid.
      gl.glColor3f(0.5, 0.5, 0.5)
      gl.glPushMatrix()
      gl.glTranslatef(0, 0.5, 0.5)
      grid()
      gl.glPopMatrix()

      # Set point size.
      w, h = self.dataloader.dimensions
      psz = max(self.window.get_size()) / float(max(
          w, h)) if self.state.scale else 1
      gl.glPointSize(psz)
      distance = (0, 0, 1) if self.state.attenuation else (1, 0, 0)
      gl.glPointParameterfv(gl.GL_POINT_DISTANCE_ATTENUATION,
                            (gl.GLfloat * 3)(*distance))

      # Set lighting.
      if self.state.lighting:
        if self.load_mano and not self.state.model_off:
          ldir = [0.0, 0.0, -1.0]
        else:
          ldir = np.dot(self.state.rotation, (0, 0, 1))
        ldir = list(ldir) + [0]
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 3)(1.0, 1.0,
                                                                   1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 3)(0.75, 0.75,
                                                                   0.75))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (gl.GLfloat * 3)(0, 0, 0))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

      gl.glColor3f(1, 1, 1)

      # Draw point cloud for each camera.
      for c in range(len(self.pcd_image)):
        if self.state.camera_off[c]:
          continue

        # Set texture matrix stack.
        gl.glMatrixMode(gl.GL_TEXTURE)
        gl.glLoadIdentity()
        gl.glTranslatef(0.5 / self.pcd_image[c].width,
                        0.5 / self.pcd_image[c].height, 0)
        tw, th = self.pcd_image[c].texture.owner.width, self.pcd_image[
            c].texture.owner.height
        gl.glScalef(self.pcd_image[c].width / float(tw),
                    self.pcd_image[c].height / float(th), 1)

        # Draw vertices and textures.
        texture = self.pcd_image[c].get_texture()
        gl.glEnable(texture.target)
        gl.glBindTexture(texture.target, texture.id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST)

        gl.glEnable(gl.GL_POINT_SPRITE)

        if not self.state.scale and not self.state.attenuation:
          gl.glDisable(gl.GL_MULTISAMPLE)
        self.pcd_vlist[c].draw(gl.GL_POINTS)
        gl.glDisable(texture.target)
        if not self.state.scale and not self.state.attenuation:
          gl.glEnable(gl.GL_MULTISAMPLE)

        # draw axes for markers
        meta = self.pcd_meta[c]
        if meta is not None:
          RT = np.eye(4, dtype=np.float32)
          for key in meta:
            if 'ar_marker' in key:
              pose = meta[key].flatten()
              RT[:3, :3] = quat2mat(pose[3:])
              RT[:3, 3] = pose[:3]
              axes(size=0.1, width=3, pose=RT)
            elif key == 'center':
              axes(size=0.1, width=3, pose=meta[key])

      gl.glDisable(gl.GL_LIGHTING)

      # Draw frustum and axes.
      gl.glColor3f(0.25, 0.25, 0.25)
      frustum(self.dataloader.dimensions, self.dataloader.master_intrinsics)
      # axes()

      # Reset matrix stacks.
      gl.glMatrixMode(gl.GL_PROJECTION)
      gl.glLoadIdentity()
      gl.glOrtho(0, width, 0, height, -1, 1)
      gl.glMatrixMode(gl.GL_MODELVIEW)
      gl.glLoadIdentity()
      gl.glMatrixMode(gl.GL_TEXTURE)
      gl.glLoadIdentity()

      gl.glDisable(gl.GL_DEPTH_TEST)

      self.fps_display.draw()

  def update(self):

    def copy(dst, src):
      """Copies numpy array to pyglet array."""
      np.array(dst, copy=False)[:] = src.ravel()

    pcd_rgb = self.dataloader.pcd_rgb
    pcd_vert = self.dataloader.pcd_vert
    pcd_tex_coord = self.dataloader.pcd_tex_coord
    pcd_mask = self.dataloader.pcd_mask
    pcd_meta = self.dataloader.pcd_meta
    
    for c in range(len(self.pcd_image)):
      if self.state.mask > 0:
        pcd_vert[c] = pcd_vert[c].copy()
        if self.state.mask == 1:
          for i in range(3):
            pcd_vert[c][:, :, i][np.logical_not(pcd_mask[c])] = 0

      self.pcd_image[c].set_data('RGB', pcd_rgb[c].strides[0],
                                 pcd_rgb[c].ctypes.data)
      copy(self.pcd_vlist[c].vertices, pcd_vert[c])
      copy(self.pcd_vlist[c].tex_coords, pcd_tex_coord[c])

      if self.state.lighting:
        dy, dx = np.gradient(pcd_vert[c], axis=(0, 1))
        n = np.cross(dx, dy)
        copy(self.pcd_vlist[c].normals, n)

      self.pcd_meta[c] = pcd_meta[c]

    if self.state.paused:
      return

    self.dataloader.step()
