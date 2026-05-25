import pathlib
import threading
import time
from datetime import datetime

import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from OpenGL import GL as gl

from gaussian_splatting.gaussian_renderer import render_pinhole, render_spherical
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gui import video_writer
from gui.gl_render import util, util_gau
from gui.gl_render.render_ogl import OpenGLRenderer
from gui.gui_utils import (
    GaussianPacket,
    Packet_vis2main,
    ParamsGUI,
    clear_queue,
    create_frustum,
    cv_gl,
    get_latest_queue,
)
from utils.camera_utils import Camera
from utils.logging_utils import Log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class SLAM_GUI:
    def __init__(self, params_gui: ParamsGUI | None = None):
        self.step = 0
        self.rendered_frames = 0
        self.process_finished = False
        self.device = "cuda"
        # If True the GUI will quit automatically when a finish signal is received.
        self.exit_gui_on_finish = True
        self.control_panel_visible = True
        self.fly_move_step = 0.05
        self._arrow_keys_pressed: set[str] = set()

        self.frustum_dict = {}
        self.model_dict = {}

        self.init_widget()

        self.q_main2vis = None
        self.q_vis2main = None
        self.gaussian_cur: GaussianPacket | None = None
        self.pipe = None
        bg_color = [0, 0, 0]
        self.background: torch.Tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.init = False
        self.kf_window = None
        self.render_img = None
        self.active_kf_ids = set()
        self.save_path = pathlib.Path(".")

        if params_gui is not None:
            self.background = params_gui.background
            # self.gaussian_cur = params_gui.gaussians
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.pipe = params_gui.pipe
            self.exit_gui_on_finish = params_gui.exit_gui_on_finish
            self.save_path = pathlib.Path(params_gui.output_dir)

        self.gaussian_nums = []

        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        # gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)
        
        self.video_writer_gui = None
        self.video_writer_3dgs = None
        self.frames_path_gui = None
        self.frames_path_3dgs = None

        threading.Thread(target=self._update_thread).start()

    def init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "ODGS-SLAM", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.window.set_on_key(self._on_key)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.button = gui.ToggleSwitch("Pause/Play")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.gui_active_chbox = gui.Checkbox("GUI active")
        self.gui_active_chbox.checked = True
        self.gui_active_chbox.set_on_checked(self._on_gui_active_chbox)
        self.panel.add_child(self.gui_active_chbox)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        # Check boxes
        vp_subtile1.add_child(gui.Label("Camera follow options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = True
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)
        vp_subtile1.add_child(chbox_tile)

        # Combo panels
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))

        # Jump to the camera viewpoint
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(gui.Label("Viewpoint list"))
        combo_tile.add_child(self.combo_kf)
        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.kf_window_chbox = gui.Checkbox("Active window")
        self.kf_window_chbox.set_on_checked(self._on_kf_window_chbox)
        chbox_tile_3dobj.add_child(self.kf_window_chbox)
        self.panel.add_child(chbox_tile_3dobj)

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = False
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        self.gt_points_chbox = gui.Checkbox("GT Points")
        self.gt_points_chbox.checked = False
        self.gt_points_chbox.set_on_checked(self._on_gt_points_chbox)
        chbox_tile_3dobj.add_child(self.gt_points_chbox)

        self.panel.add_child(gui.Label("Camera model"))
        chbox_tile_cam_model = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.spherical_chbox = gui.Checkbox("Use Spherical Rendering")
        self.spherical_chbox.checked = False
        self.spherical_chbox.set_on_checked(self._on_spherical_chbox)
        chbox_tile_cam_model.add_child(self.spherical_chbox)
        self.panel.add_child(chbox_tile_cam_model)

        self.panel.add_child(gui.Label("Rendering options"))
        chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        chbox_tile_geometry.add_child(self.depth_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        chbox_tile_geometry.add_child(self.opacity_chbox)

        self.time_shader_chbox = gui.Checkbox("Time Shader")
        self.time_shader_chbox.checked = False
        chbox_tile_geometry.add_child(self.time_shader_chbox)

        self.elipsoid_chbox = gui.Checkbox("Elipsoid Shader")
        self.elipsoid_chbox.checked = False
        chbox_tile_geometry.add_child(self.elipsoid_chbox)
        self.panel.add_child(chbox_tile_geometry)

        slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        slider_label = gui.Label("Gaussian Scale (0-1)")
        self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scaling_slider.set_limits(0.001, 1.0)
        self.scaling_slider.double_value = 1.0
        slider_tile.add_child(slider_label)
        slider_tile.add_child(self.scaling_slider)
        self.panel.add_child(slider_tile)

        fly_speed_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        fly_speed_label = gui.Label("Fly Speed")
        self.fly_speed_slider = gui.Slider(gui.Slider.DOUBLE)
        self.fly_speed_slider.set_limits(0.001, 0.1)
        self.fly_speed_slider.double_value = self.fly_move_step
        fly_speed_tile.add_child(fly_speed_label)
        fly_speed_tile.add_child(self.fly_speed_slider)
        self.panel.add_child(fly_speed_tile)

        # screenshot buttom
        self.panel.add_child(gui.Label("Capture Options"))
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        video_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.capture_video_switch = gui.ToggleSwitch("Capture video: Off/On")
        self.capture_video_switch.is_on = False
        self.capture_video_switch.set_on_clicked(self._on_capture_video_switch)
        video_tile.add_child(self.capture_video_switch)
        self.video_output_type_chbox = gui.Checkbox("Output video")
        self.video_output_type_chbox.checked = True
        video_tile.add_child(self.video_output_type_chbox)
        self.images_output_type_chbox = gui.Checkbox("Output images")
        self.images_output_type_chbox.checked = False
        video_tile.add_child(self.images_output_type_chbox)
        self.panel.add_child(video_tile)

        self.panel.add_child(gui.Label("Capture target"))
        video_target_chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.capture_with_gui_chbox = gui.Checkbox("Capture with GUI")
        self.capture_with_gui_chbox.checked = False
        video_target_chbox_tile.add_child(self.capture_with_gui_chbox)
        self.capture_with_3dgs_chbox = gui.Checkbox("Capture 3DGS only")
        self.capture_with_3dgs_chbox.checked = False
        video_target_chbox_tile.add_child(self.capture_with_3dgs_chbox)
        self.panel.add_child(video_target_chbox_tile)
        

        # Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)
        self.output_info = gui.Label("Number of Gaussians: ")
        tab_info.add_child(self.output_info)

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Input Color/Depth"))
        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)
        tab_info.add_child(gui.Label("Press 'H' to hide/show control panel."))

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(-4)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    # def add_camera(self, camera, name, color=[0, 1, 0], gt=False, radius=1.0, num_points=100):
        #     W2C = (
        #         getWorld2View2(camera.R_gt, camera.T_gt)
        #         if gt
        #         else getWorld2View2(camera.R, camera.T)
        #     )
        #     W2C = W2C.cpu().numpy()
        #     C2W = np.linalg.inv(W2C)
        #     frustum = create_spherical_frustum(C2W, color, radius=radius, num_points=num_points)
        #     if name not in self.frustum_dict.keys():
        #         frustum = create_spherical_frustum(C2W, color, radius=radius, num_points=num_points)
    def add_camera(self, camera, name, color=None, gt=False, size=0.01):
        if color is None:
            color = [0, 1, 0]
        W2C = (
            getWorld2View2(camera.R_gt, camera.T_gt)
            if gt
            else getWorld2View2(camera.R, camera.T)
        )
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict:
            frustum = create_frustum(C2W, color)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(
            name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def add_gt_point(self, camera, name):
        W2C_gt = getWorld2View2(camera.R_gt, camera.T_gt)
        W2C_gt = W2C_gt.cpu().numpy()
        C2W_gt = np.linalg.inv(W2C_gt)
        # add gt point
        gt_name = f"{name}_gt"
        gt_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        gt_point.paint_uniform_color([1, 0, 0])
        gt_point.translate(C2W_gt[:3, 3])
        self.widget3d.scene.add_geometry(gt_name, gt_point, self.lit_geo)
        self.widget3d.scene.show_geometry(gt_name, self.gt_points_chbox.checked)
        
        # add line from camera to gt point
        start = gt_point.get_center()
        end = self.frustum_dict[name].view_dir[1]
        if np.linalg.norm(start - end) > 0.002:
            points = [start, end]
            lines = [[0, 1]]
            colors = [[1, 0, 0]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            edge_name = f"{name}_gt_line"
            self.widget3d.scene.add_geometry(edge_name, line_set, self.lit)
            self.widget3d.scene.show_geometry(edge_name, self.gt_points_chbox.checked)


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        if self.control_panel_visible:
            self.widget3d_width_ratio = 0.7
        else:
            self.widget3d_width_ratio = 1.0
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        if self.control_panel_visible:
            self.panel.frame = gui.Rect(
                self.widget3d.frame.get_right(),
                contentRect.y,
                contentRect.width - self.widget3d_width,
                contentRect.height,
            )

    def _on_close(self):
        if self.q_vis2main is not None:
            packet = Packet_vis2main(flag_gui_exit=True)
            self.q_vis2main.put(packet)
        self.q_main2vis = None
        self.q_vis2main = None
        self.process_finished = True
        return True  # False would cancel the close
    
    def _on_key(self, event):
        key_to_dir = {
            gui.KeyName.UP: "forward",
            gui.KeyName.DOWN: "backward",
            gui.KeyName.LEFT: "left",
            gui.KeyName.RIGHT: "right",
        }

        if event.key in key_to_dir:
            direction = key_to_dir[event.key]
            key_repeat = getattr(gui.KeyEvent, "REPEAT", None)
            if event.type == gui.KeyEvent.DOWN or (
                key_repeat is not None and event.type == key_repeat
            ):
                self._arrow_keys_pressed.add(direction)
            elif event.type == gui.KeyEvent.UP:
                self._arrow_keys_pressed.discard(direction)

        if event.key == gui.KeyName.H and event.type == gui.KeyEvent.DOWN:
            self.control_panel_visible = not self.control_panel_visible
            self.panel.visible = self.control_panel_visible
            self._on_layout(None)
            Log(f"Toggle control panel visibile: {self.control_panel_visible}", tag="GUI")
            
        if event.key == gui.KeyName.C and event.type == gui.KeyEvent.DOWN:
            self.capture_video_switch.is_on = not self.capture_video_switch.is_on
            self._on_capture_video_switch(self.capture_video_switch.is_on)
            Log(f"Toggle video capture: {self.capture_video_switch.is_on}", tag="GUI")

    def _apply_arrow_key_camera_motion(self):
        if not self._arrow_keys_pressed:
            return
        # Keep follow-camera controls authoritative when enabled.
        if self.followcam_chbox.checked:
            return

        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()
        c2w = np.linalg.inv(w2c)

        eye = c2w[:3, 3].astype(np.float64)
        rot = c2w[:3, :3].astype(np.float64)

        forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = rot @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)

        if np.linalg.norm(forward) < 1e-8 or np.linalg.norm(up) < 1e-8 or np.linalg.norm(right) < 1e-8:
            return

        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)
        right = right / np.linalg.norm(right)

        move = np.zeros(3, dtype=np.float64)
        if "forward" in self._arrow_keys_pressed:
            move += forward
        if "backward" in self._arrow_keys_pressed:
            move -= forward
        if "left" in self._arrow_keys_pressed:
            move -= right
        if "right" in self._arrow_keys_pressed:
            move += right

        move_norm = np.linalg.norm(move)
        if move_norm < 1e-8:
            return

        speed = self.fly_speed_slider.double_value
        move = move / move_norm
        eye = eye + move * speed
        center = eye + forward

        self.widget3d.look_at(center, eye, up)


    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_axis_chbox(self, is_checked):
        name = "axis"
        if is_checked:
            self.widget3d.scene.remove_geometry(name)
            self.widget3d.scene.add_geometry(name, self.axis, self.lit_geo)
        else:
            self.widget3d.scene.remove_geometry(name)

    def _on_kf_window_chbox(self, is_checked):
        if self.kf_window is None:
            return
        edge_cnt = 0
        for key in self.kf_window:
            for kf_idx in self.kf_window[key]:
                name = f"kf_edge_{edge_cnt}"
                edge_cnt += 1
                if f"keyframe_{key}" not in self.frustum_dict:
                    continue
                test1 = self.frustum_dict[f"keyframe_{key}"].view_dir[1]
                kf = self.frustum_dict[f"keyframe_{kf_idx}"].view_dir[1]
                points = [test1, kf]
                lines = [[0, 1]]
                colors = [[0, 1, 0]]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                if is_checked:
                    self.widget3d.scene.remove_geometry(name)
                    self.widget3d.scene.add_geometry(name, line_set, self.lit)
                else:
                    self.widget3d.scene.remove_geometry(name)

    def _on_button(self, is_on):
        if self.q_vis2main is not None:
            packet = Packet_vis2main(flag_pause=not self.button.is_on)
            self.q_vis2main.put(packet)

    def _on_screenshot_btn(self):
        if self.render_img is None:
            return
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = pathlib.Path(self.save_path) / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "screenshot"
        current_cam = self.get_current_cam()
        if (
            self.gaussian_cur is not None
            and self.gaussian_cur.gaussians is not None
            and self.gaussian_cur.current_frame is not None
        ):
            current_cam.image_height = self.gaussian_cur.current_frame.image_height
            current_cam.image_width = self.gaussian_cur.current_frame.image_width
            with torch.no_grad():
                results = render_spherical(
                    current_cam,
                    self.gaussian_cur.gaussians,
                    self.pipe,
                    self.background,
                    self.scaling_slider.double_value,
                    mapping_mode=False,
                    tracking_mode=False
                )
            if results is None:
                Log("Rendering failed, could not save screenshot!", tag="GUI")
                return
            # Save RGB
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{filename}-spherical.png", rgb)
            # Save Depth
            if "depth" in results:
                depth = results["depth"][0].detach().cpu().numpy()
                max_depth = np.max(depth)
                depth_vis = imgviz.depth2rgb(
                    depth, min_value=0.1, max_value=max_depth, colormap="turbo"
                )
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{filename}-spherical-depth.png", depth_vis)
    
    def _on_capture_video_switch(self, is_on):
        if is_on:
            if self.capture_with_gui_chbox.checked is False and self.capture_with_3dgs_chbox.checked is False:
                Log("Please select at least one capture target (with GUI or 3DGS)!", tag="GUI")
                self.capture_video_switch.is_on = False
            if self.video_output_type_chbox.checked is False and self.images_output_type_chbox.checked is False:
                Log("Please select at least one output type (video or images)!", tag="GUI")
                self.capture_video_switch.is_on = False
            if self.capture_video_switch.is_on: 
                self.video_output_type_chbox.enabled = False
                self.capture_with_3dgs_chbox.enabled = False
                self.capture_with_gui_chbox.enabled = False
        else:
            self.video_writer_gui = None
            self.video_writer_3dgs = None
            self.frames_path_gui = None
            self.frames_path_3dgs = None
            self.video_output_type_chbox.enabled = True
            self.capture_with_3dgs_chbox.enabled = True
            self.capture_with_gui_chbox.enabled = True

    def _on_gt_points_chbox(self, is_checked):
        for name in self.frustum_dict:
            gt_name = f"{name}_gt"
            if self.widget3d.scene.has_geometry(gt_name):
                self.widget3d.scene.show_geometry(gt_name, is_checked)
            edge_name = f"{name}_gt_line"
            if self.widget3d.scene.has_geometry(edge_name):
                self.widget3d.scene.show_geometry(edge_name, is_checked)

    def _on_spherical_chbox(self, is_checked):
        if is_checked:
            self.elipsoid_chbox.checked = False
            self.elipsoid_chbox.enabled = False
        else:
            self.elipsoid_chbox.enabled = True
            
    def _on_gui_active_chbox(self, is_checked):
        if self.q_vis2main is not None:
            self.q_vis2main.put(Packet_vis2main(flag_gui_active=is_checked))
        if not is_checked:
            self._on_axis_chbox(is_checked=False)
            self._on_cameras_chbox(is_checked=False)
            self._on_kf_window_chbox(is_checked=False)
            self._on_gt_points_chbox(is_checked=False)
            self.axis_chbox.enabled = False
            self.cameras_chbox.enabled = False
            self.kf_window_chbox.enabled = False
            self.gt_points_chbox.enabled = False
        else:
            self.axis_chbox.enabled = True
            self.cameras_chbox.enabled = True
            self.kf_window_chbox.enabled = True
            self.gt_points_chbox.enabled = True
            self._on_axis_chbox(is_checked=self.axis_chbox.checked)
            self._on_cameras_chbox(is_checked=self.cameras_chbox.checked)
            self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)
            self._on_gt_points_chbox(is_checked=self.gt_points_chbox.checked)


    # @staticmethod
    # def resize_img(img, width):
    #     height = int(width * img.shape[0] / img.shape[1])
    #     return cv2.resize(img, (width, height))

    # def add_ids(self):
    #     indices = (
    #         torch.unique(
    #             self.gaussian_cur.unique_kfIDs).cpu().numpy().astype(int)
    #     ).tolist()
    #     for idx in indices:
    #         if idx in self.gaussian_id_dict.keys():
    #             continue

    #         self.gaussian_id_dict[idx] = 0
    #         self.combo_gaussian_id.add_item(str(idx))
    
    def write_rendered_image(self, frame):
        if self.render_img is None:
            return

        save_dir = self.save_path / "video"
        save_dir.mkdir(parents=True, exist_ok=True)
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance

        if self.capture_with_gui_chbox.checked:
            img_ui = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
            img_ui = cv2.cvtColor(img_ui, cv2.COLOR_BGR2RGB)

            if self.video_output_type_chbox.checked:
                if self.video_writer_gui is None:
                    self.video_writer_gui = video_writer.VideoWriter(save_dir / f"video_gui_start_frame{frame:06d}", img_ui.shape[1], img_ui.shape[0], fps=30)
                try:
                    self.video_writer_gui.write_frame(img_ui)
                except Exception as e:
                    Log(f"Error writing frame to video_writer_gui, resetting: {e}", tag="GUI")
                    self.video_writer_gui = None

            if self.images_output_type_chbox.checked and self.rendered_frames % 10 == 0:
                if self.frames_path_gui is None:
                    self.frames_path_gui = save_dir / f"frames_gui_start_frame{frame:06d}"
                    self.frames_path_gui.mkdir(parents=True, exist_ok=True)
                filename = self.frames_path_gui / f"{frame:06d}"
                cv2.imwrite(f"{filename}-gui.jpg", img_ui)

        if self.capture_with_3dgs_chbox.checked:
            img = np.asarray(self.render_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.video_output_type_chbox.checked:
                if self.video_writer_3dgs is None:
                    self.video_writer_3dgs = video_writer.VideoWriter(save_dir / f"video_3dgs_start_frame{frame:06d}", img.shape[1], img.shape[0], fps=30)
                try:
                    self.video_writer_3dgs.write_frame(img)
                except Exception as e:
                    Log(f"Error writing frame to video_writer_3dgs, resetting: {e}", tag="GUI")
                    self.video_writer_3dgs = None

            if self.images_output_type_chbox.checked and self.rendered_frames % 10 == 0:
                if self.frames_path_3dgs is None:
                    self.frames_path_3dgs = save_dir / f"frames_3dgs_start_frame{frame:06d}"
                    self.frames_path_3dgs.mkdir(parents=True, exist_ok=True)
                filename = self.frames_path_3dgs / f"{frame:06d}"
                cv2.imwrite(f"{filename}.jpg", img)
                
    def receive_data(self, q):
        if q is None or self.gui_active_chbox.checked is False:
            return False

        gaussian_packet = get_latest_queue(q)
        if gaussian_packet is None:
            return False

        update_received = False
        if gaussian_packet.gaussians is not None:
            self.gaussian_cur = gaussian_packet
            self.output_info.text = f"Number of Gaussians: {self.gaussian_cur.gaussians.get_xyz.shape[0]}"
            self.init = True
            update_received = True

        if gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
            update_received = True

        if gaussian_packet.keyframes is not None:
            for keyframe in gaussian_packet.keyframes:
                name = f"keyframe_{keyframe.uid}"
                frustum = self.add_camera(keyframe, name=name, color=[0, 0, 1])
                self.add_gt_point(keyframe, name=name)
            update_received = True

        if gaussian_packet.kf_window is not None:
            self.kf_window = gaussian_packet.kf_window
            self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)
            update_received = True

        if gaussian_packet.gtcolor is not None:
            if isinstance(gaussian_packet.gtcolor, np.ndarray):
                rgb = np.clip(gaussian_packet.gtcolor, 0, 1.0) * 255
                rgb = rgb.astype(np.uint8)
            else:
                rgb = torch.clamp(gaussian_packet.gtcolor, min=0, max=1.0) * 255
                rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)
            update_received = True

        if gaussian_packet.gtdepth is not None:
            if isinstance(gaussian_packet.gtdepth, torch.Tensor):
                depth = gaussian_packet.gtdepth.detach().cpu().numpy()
            else:
                depth = gaussian_packet.gtdepth
            min_val = float(np.min(depth))
            max_val = float(np.max(depth))
            depth = imgviz.depth2rgb(
                depth, min_value=min_val, max_value=max_val, colormap="jet"
            )
            rgb = o3d.geometry.Image(depth)
            self.in_depth_widget.update_image(rgb)
            update_received = True

        if gaussian_packet.finish:
            Log("Received terminate signal", tag="GUI")
            # clean up the pipe
            clear_queue(self.q_vis2main)
            clear_queue(self.q_main2vis)
            self.q_vis2main = None
            self.q_main2vis = None
            if self.exit_gui_on_finish:
                self.process_finished = True
            update_received = True
            
        if hasattr(gaussian_packet, "active_kf_ids") and gaussian_packet.active_kf_ids is not None:
            new_active_kf_ids = set(gaussian_packet.active_kf_ids)
            inactive_kf_ids = self.active_kf_ids - new_active_kf_ids
            for kf_id in inactive_kf_ids:
                name = f"keyframe_{kf_id}"
                if name in self.frustum_dict:
                    frustum = self.frustum_dict[name]
                    pose = frustum.pose if hasattr(frustum, "pose") else None
                    if pose is not None:
                        red_frustum = create_frustum(pose, frusutum_color=[1, 0, 0])
                        self.frustum_dict[name] = red_frustum
                        self.widget3d.scene.remove_geometry(name)
                        self.widget3d.scene.add_geometry(name, red_frustum.line_set, self.lit)
                        self.widget3d.scene.set_geometry_transform(name, pose.astype(np.float64))
                        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
            self.active_kf_ids = new_active_kf_ids
            update_received = True
        return update_received

    # @staticmethod
    # def depth_to_normal(points, k=3, d_min=1e-3, d_max=10.0):
    #     k = (k - 1) // 2
    #     # points: (B, 3, H, W)
    #     b, _, h, w = points.size()
    #     points_pad = F.pad(
    #         points, (k, k, k, k), mode="constant", value=0
    #     )  # (B, 3, k+H+k, k+W+k)
    #     if d_max is not None:
    #         valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
    #             points_pad[:, 2:, :, :] < d_max
    #         )  # (B, 1, k+H+k, k+W+k)
    #     else:
    #         valid_pad = points_pad[:, 2:, :, :] > d_min
    #     valid_pad = valid_pad.float()

    #     # vertical vector (top - bottom)
    #     vec_vert = (
    #         points_pad[:, :, :h, k: w + k]
    #         - points_pad[:, :, 2 * k: h + (2 * k), k: w + k]
    #     )

    #     # horizontal vector (left - right)
    #     vec_hori = (
    #         points_pad[:, :, k: h + k, :w]
    #         - points_pad[:, :, k: h + k, 2 * k: w + (2 * k)]
    #     )

    #     # valid_mask
    #     valid_mask = (
    #         valid_pad[:, :, k: h + k, k: w + k]
    #         * valid_pad[:, :, :h, k: w + k]
    #         * valid_pad[:, :, 2 * k: h + (2 * k), k: w + k]
    #         * valid_pad[:, :, k: h + k, :w]
    #         * valid_pad[:, :, k: h + k, 2 * k: w + (2 * k)]
    #     )
    #     valid_mask = valid_mask > 0.5

    #     # get cross product (B, 3, H, W)
    #     cross_product = -torch.linalg.cross(vec_vert, vec_hori, dim=1)
    #     normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
    #     return normal, valid_mask

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(
            vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        T = torch.from_numpy(w2c)
        current_cam = Camera.init_from_gui(
            uid=-1,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=image_gui.shape[1],
            W=image_gui.shape[2],
        )
        current_cam.update_RT(T[0:3, 0:3], T[0:3, 3])
        return current_cam

    def rasterise(self, current_cam):
        if self.gaussian_cur is None or self.gaussian_cur.gaussians is None:
            return None

        if (self.time_shader_chbox.checked):
            features = self.gaussian_cur.gaussians.get_features.clone()
            kf_ids = self.gaussian_cur.gaussians.unique_kfIDs.float()
            rgb_kf = imgviz.depth2rgb(
                kf_ids.view(-1, 1).cpu().numpy(), colormap="jet", dtype=np.float32
            )
            alpha = 0.1
            self.gaussian_cur.gaussians.get_features = alpha * features + (
                1 - alpha
            ) * torch.from_numpy(rgb_kf).to(features.device)
            with torch.no_grad():
                if self.spherical_chbox.checked:
                    rendering_data = render_spherical(
                        current_cam,
                        self.gaussian_cur.gaussians,
                        self.pipe,
                        self.background,
                        self.scaling_slider.double_value,
                        mapping_mode=False,
                        tracking_mode=False,
                    )
                else:
                    rendering_data = render_pinhole(
                        current_cam,
                        self.gaussian_cur.gaussians,
                        self.pipe,
                        self.background,
                        self.scaling_slider.double_value,
                    )
            self.gaussian_cur.gaussians.get_features = features
        else:
            with torch.no_grad():
                if self.spherical_chbox.checked:
                    rendering_data = render_spherical(
                        current_cam,
                        self.gaussian_cur.gaussians,
                        self.pipe,
                        self.background,
                        self.scaling_slider.double_value,
                        tracking_mode=False,
                        mapping_mode=False,
                    )
                else:
                    rendering_data = render_pinhole(
                        current_cam,
                        self.gaussian_cur.gaussians,
                        self.pipe,
                        self.background,
                        self.scaling_slider.double_value,
                    )

        return rendering_data

    def render_o3d_image(self, results, current_cam):
        if results is None:
            # return black image if no results
            img = np.zeros([current_cam.image_height, current_cam.image_width, 3], dtype=np.uint8)
            render_img = o3d.geometry.Image(img)
        elif self.depth_chbox.checked:
            depth = results["depth"]
            depth = depth[0, :, :].detach().cpu().numpy()
            max_depth = np.max(depth)
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=max_depth, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(depth)

        elif self.opacity_chbox.checked:
            opacity = results["opacity"]
            if len(opacity.shape) == 2:
                opacity = opacity.detach().cpu().numpy()
            else:
                opacity = opacity[0, :, :].detach().cpu().numpy()
            max_opacity = np.max(opacity)
            opacity = imgviz.depth2rgb(
                opacity, min_value=0.0, max_value=max_opacity, colormap="jet"
            )
            opacity = torch.from_numpy(opacity)
            opacity = torch.permute(opacity, (2, 0, 1)).float()
            opacity = (opacity).byte().permute(
                1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(opacity)

        elif self.elipsoid_chbox.checked:
            if self.gaussian_cur is None or self.gaussian_cur.gaussians is None:
                return None
            glfw.poll_events()
            gl.glClearColor(0, 0, 0, 1.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w = int(self.window.size.width * self.widget3d_width_ratio)
            glfw.set_window_size(self.window_gl, w, self.window.size.height)
            self.g_camera.fovy = current_cam.FoVy
            self.g_camera.update_resolution(self.window.size.height, w)
            self.g_renderer.set_render_reso(w, self.window.size.height)
            frustum = create_frustum(
                np.linalg.inv(
                    cv_gl @ self.widget3d.scene.camera.get_view_matrix())
            )

            self.g_camera.position = frustum.eye.astype(np.float32)
            self.g_camera.target = frustum.center.astype(np.float32)
            self.g_camera.up = frustum.up.astype(np.float32)

            self.gaussians_gl.xyz = self.gaussian_cur.gaussians.get_xyz.cpu().numpy()
            self.gaussians_gl.opacity = self.gaussian_cur.gaussians.get_opacity.cpu().numpy()
            self.gaussians_gl.scale = self.gaussian_cur.gaussians.get_scaling.cpu().numpy()
            self.gaussians_gl.rot = self.gaussian_cur.gaussians.get_rotation.cpu().numpy()
            self.gaussians_gl.sh = self.gaussian_cur.gaussians.get_features.cpu().numpy()[
                :, 0, :]

            self.update_activated_renderer_state(self.gaussians_gl)
            self.g_renderer.sort_and_update(self.g_camera)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.g_renderer.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            cv2.flip(img, 0)
            render_img = o3d.geometry.Image(img)
            glfw.swap_buffers(self.window_gl)
        else:
            rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)
        return render_img

    def render_gui(self):
        if not self.init:
            return
        current_cam = self.get_current_cam()
        results = None
        if self.gui_active_chbox.checked:
            results = self.rasterise(current_cam)
        self.render_img = self.render_o3d_image(results, current_cam) 
        self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)

    def scene_update(self):
        self.rendered_frames += 1
        self.receive_data(self.q_main2vis)
        # self._apply_arrow_key_camera_motion()
        self.render_gui()
        if (self.capture_video_switch.is_on):
            self.write_rendered_image(self.rendered_frames)

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                # stop video writers if running
                if self.video_writer_gui is not None:
                    self.video_writer_gui.end_recording()
                if self.video_writer_3dgs is not None:
                    self.video_writer_3dgs.end_recording()
                Log("Closing Visualization", tag="GUI")
                break

            def update():
                # Apply keyboard translation on every UI tick so it stays smooth
                # while the user is also rotating/panning with the mouse.
                self._apply_arrow_key_camera_motion()

                if self.step % 3 == 0:
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)
        Log("Visualization thread ended", tag="GUI")


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()
    Log("Application run finished", tag="GUI")


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
