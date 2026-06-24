from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from argclz import AbstractParser, argument, validator
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import SLICE_DIMENSION_10um
from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import get_slice_view
from neuralib.imglib.transform import apply_transformation
from neuralib.util.verbose import fprint, print_save

__all__ = ['RegisterOptions', 'estimate_transform', 'save_transform']


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """rotate about the image center, keeping the original shape"""
    if not deg:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def _to_u8(img: np.ndarray, contrast: tuple[float, float] | None = None) -> np.ndarray:
    """map to uint8 using the given (lo, hi) contrast window, else full min-max"""
    img = np.asarray(img, dtype=float)
    lo, hi = contrast if contrast is not None else (float(img.min()), float(img.max()))
    return (np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1) * 255).astype(np.uint8)


def estimate_transform(slice_xy: np.ndarray, atlas_xy: np.ndarray, *, affine: bool = False) -> np.ndarray:
    """Estimate the 3x3 matrix mapping histology (slice) points onto atlas points.

    Matches ``apply_transformation`` / ``cv2.warpPerspective`` convention: the returned
    matrix warps the slice image into atlas space.

    :param slice_xy: ``Array[float, [N, 2]]`` (x, y) points on the resized histology slice.
    :param atlas_xy: ``Array[float, [N, 2]]`` (x, y) matched points on the atlas plane.
    :param affine: estimate an affine (6 DOF) instead of projective (8 DOF) transform.
    :return: ``Array[float64, [3, 3]]``
    """
    slice_xy = np.asarray(slice_xy, dtype=np.float64)
    atlas_xy = np.asarray(atlas_xy, dtype=np.float64)
    if slice_xy.shape != atlas_xy.shape:
        raise ValueError(f'point count mismatch: {slice_xy.shape} vs {atlas_xy.shape}')

    if affine:
        if len(slice_xy) < 3:
            raise ValueError('affine transform needs >=3 matched point pairs')
        m, _ = cv2.estimateAffine2D(slice_xy, atlas_xy)
        return np.vstack([m, [0, 0, 1]]).astype(np.float64)
    else:
        if len(slice_xy) < 4:
            raise ValueError('projective transform needs >=4 matched point pairs')
        m, _ = cv2.findHomography(slice_xy, atlas_xy)
        return m.astype(np.float64)


def save_transform(matrix: np.ndarray, *,
                   output_dir: Path, name: str,
                   plane: PLANE_TYPE, resolution: int,
                   slice_index: int, dw: int, dh: int,
                   slice_xy: np.ndarray, atlas_xy: np.ndarray,
                   rotate: float = 0.0, flip_lr: bool = False, flip_ud: bool = False,
                   contrast: tuple[float, float] | None = None) -> Path:
    """Save the 3x3 matrix and metadata into a single ``.json``. Returns its path.

    ``rotate``/``flip_lr``/``flip_ud`` record the preprocessing so the result can be
    reproduced (raw -> flip -> rotate -> resize -> apply matrix) and the session resumed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'matrix': np.asarray(matrix, dtype=float).tolist(),
        'plane': plane,
        'resolution': resolution,
        'slice_index': int(slice_index),
        'dw': int(dw),
        'dh': int(dh),
        'rotate': float(rotate),
        'flip_lr': bool(flip_lr),
        'flip_ud': bool(flip_ud),
        'contrast': list(contrast) if contrast is not None else None,
        'slice_xy': np.asarray(slice_xy, dtype=float).tolist(),
        'atlas_xy': np.asarray(atlas_xy, dtype=float).tolist(),
    }
    js = output_dir / f'{name}_transform.json'
    js.write_text(json.dumps(meta, indent=2))
    print_save(js)
    return js


class RegisterOptions(AbstractParser):
    DESCRIPTION = 'Interactively register a histology slice to the Allen CCF (napari)'

    raw_image: Path = argument(
        '-I', '--image',
        validator=validator.path.is_exists(),
        help='histology image path (registered in atlas space)'
    )

    cut_plane: PLANE_TYPE = argument(
        '-P', '--plane-type',
        default='coronal',
        help='cutting orientation'
    )

    resolution: int = argument('--resolution', default=10, help='atlas resolution (um)')

    output_dir: Path | None = argument(
        '-O', '--output-dir',
        default=None,
        help='output directory (default: <image-dir>/transformations)'
    )

    name: str | None = argument('--name', default=None, help='output name (default: image stem)')

    flip_lr: bool = argument('--flip-lr', help='flip histology left-right before registration')
    flip_ud: bool = argument('--flip-ud', help='flip histology up-down before registration')
    affine: bool = argument('--affine', help='use affine instead of projective transform')
    boundary_color: str = argument(
        '--boundary-color',
        default='orange',
        help='annotation boundary overlay color (matplotlib name or #hex)'
    )
    load: Path | None = argument(
        '--load',
        default=None,
        help='resume from a saved *_transform.json (restores points + index/dw/dh/rotate/flips)'
    )

    def run(self):
        if self.cut_plane not in SLICE_DIMENSION_10um:
            raise ValueError(f'plane {self.cut_plane!r} not supported yet '
                             f'(available: {list(SLICE_DIMENSION_10um)})')

        out_dir = self.output_dir or self.raw_image.parent / 'transformations'
        name = self.name or self.raw_image.stem

        load = json.loads(self.load.read_text()) if self.load else None
        if load:  # resume: preprocessing must match the saved session
            self.flip_lr = load.get('flip_lr', self.flip_lr)
            self.flip_ud = load.get('flip_ud', self.flip_ud)

        view = get_slice_view('reference', self.cut_plane, resolution=self.resolution)
        atlas_w = int(view.width)

        oriented = iio.imread(self.raw_image)
        if self.flip_ud:
            oriented = np.flipud(oriented)
        if self.flip_lr:
            oriented = np.fliplr(oriented)
        self._oriented = oriented  # pre-rotate, pre-resize; the rotation slider re-derives from this

        self._launch_napari(view, atlas_w, out_dir, name, load)

    # (width, height) anatomical axes per plane, for tilt labels
    AXIS = {'coronal': ('ML', 'DV'), 'sagittal': ('AP', 'DV')}

    def _launch_napari(self, ref_view, atlas_w: int, out_dir: Path, name: str, load: dict | None = None):
        import napari
        from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton, SpinBox

        load = load or {}
        dim = SLICE_DIMENSION_10um[self.cut_plane]

        def make_hist(angle: float) -> np.ndarray:
            return cv2.resize(_rotate(self._oriented, angle), dim)

        angle0 = float(load.get('rotate', 0.0))
        state = {'index': int(load.get('slice_index', int(ref_view.n_planes) // 2)),
                 'dw': int(load.get('dw', 0)), 'dh': int(load.get('dh', 0)),
                 'expect': 'atlas', 'ann': None, 'hist': make_hist(angle0)}

        ann_view = get_slice_view('annotation', self.cut_plane, resolution=self.resolution)
        structures = BrainGlobeAtlas(f'allen_mouse_{self.resolution}um').structures

        def plane_image() -> np.ndarray:
            i, dw, dh = state['index'], state['dw'], state['dh']
            od = lambda v: v + 1 if v != 0 else 0
            state['ann'] = ann_view.plane_at(i).with_offset(od(dw), od(dh)).image
            ref_plane = ref_view.plane_at(i).with_offset(od(dw), od(dh))
            state['ref_mm'] = ref_plane.reference_value
            return ref_plane.image

        def boundary_mask(ann: np.ndarray) -> np.ndarray:
            """region boundaries = pixels where the annotation id changes vs its right/down neighbour"""
            b = np.zeros(ann.shape, dtype=float)
            b[:, :-1] = np.maximum(b[:, :-1], ann[:, :-1] != ann[:, 1:])
            b[:-1, :] = np.maximum(b[:-1, :], ann[:-1, :] != ann[1:, :])
            return b

        def region_name(y: float, x: float) -> str:
            ann = state['ann']
            if ann is None or not (0 <= y < ann.shape[0] and 0 <= x < ann.shape[1]):
                return ''
            rid = int(ann[int(y), int(x)])
            if rid == 0:
                return ''
            try:
                s = structures[rid]
                return f"{s['acronym']} — {s['name']}"
            except KeyError:
                return f'id {rid}'

        def pts_kw(color):
            return dict(face_color=color, border_color=color, symbol='cross', size=20, ndim=2,
                        features={'n': np.empty(0, dtype='<U3')},
                        text={'string': '{n}', 'color': color, 'size': 12, 'translation': [-12, 0]})

        from matplotlib.colors import to_rgb
        from napari.utils import Colormap
        brgb = to_rgb(self.boundary_color)  # (r, g, b) in 0..1
        state['brgb'] = brgb
        bcmap = Colormap([[0, 0, 0], list(brgb)], name='boundary')  # 0 -> transparent (additive), 1 -> color

        viewer = napari.Viewer(title=f'ccf2d register — {name}')
        viewer.text_overlay.visible = True
        viewer.text_overlay.font_size = 18
        viewer.text_overlay.color = 'yellow'
        atlas_layer = viewer.add_image(plane_image(), name='atlas', colormap='gray')
        bound_layer = viewer.add_image(boundary_mask(state['ann']), name='boundaries',
                                       colormap=bcmap, blending='additive', opacity=0.9)
        hist_layer = viewer.add_image(state['hist'], name='histology', translate=(0, atlas_w))

        # reference xy grid over the histology (toggled off), 100 px spacing. drawn as an
        # image on the pixel grid (nearest) so it scales with the slice and is zoom-stable
        gw, gh = dim
        step = 100
        grid_img = np.zeros((gh, gw), dtype=float)
        grid_img[::step, :] = 1
        grid_img[:, ::step] = 1
        grid_img[-1, :] = grid_img[:, -1] = 1
        grid_layer = viewer.add_image(grid_img, name='xy_grid', translate=(0, atlas_w),
                                      colormap='gray', blending='additive', opacity=0.5,
                                      interpolation2d='nearest')
        grid_layer.visible = False

        atlas_pts = viewer.add_points(name='atlas_pts', **pts_kw('red'))
        slice_pts = viewer.add_points(name='slice_pts', **pts_kw('cyan'))

        def renumber(layer):
            layer.features = {'n': np.array([str(i + 1) for i in range(len(layer.data))], dtype='<U3')}

        def add_pt(layer, pos):
            layer.data = np.vstack([layer.data, pos]) if len(layer.data) else np.array([pos])
            renumber(layer)

        @viewer.mouse_drag_callbacks.append
        def on_click(_v, event):
            dragged = False
            yield
            while event.type == 'mouse_move':
                dragged = True
                yield
            if dragged or not pick_w.value:
                return
            y, x = event.position  # world coords (row, col)
            n = len(atlas_pts.data) + 1
            if state['expect'] == 'atlas':
                if x >= atlas_w:
                    status.value = f'expected an ATLAS click (left side) for pt {n}'
                    return
                add_pt(atlas_pts, (y, x))
                state['expect'] = 'slice'
                status.value = f'now click the matching point on the histology (slice pt {n})'
            else:
                if x < atlas_w:
                    status.value = f'expected a SLICE click (right side) for pt {len(slice_pts.data) + 1}'
                    return
                add_pt(slice_pts, (y, x))
                state['expect'] = 'atlas'
                status.value = f'pair {len(slice_pts.data)} set — click atlas landmark {len(slice_pts.data) + 1}'

        w_axis, h_axis = self.AXIS.get(self.cut_plane, ('w', 'h'))
        res = self.resolution
        idx_w = SpinBox(label='slice index (voxel)', value=state['index'], min=0, max=int(ref_view.n_planes) - 1)
        dw_w = SpinBox(label=f'dw / {w_axis} tilt (voxel)', value=state['dw'], min=-200, max=200)
        dh_w = SpinBox(label=f'dh / {h_axis} tilt (voxel)', value=state['dh'], min=-200, max=200)
        rot_w = SpinBox(label='rotate (deg)', value=angle0, min=-180, max=180)
        pick_w = CheckBox(label='pick points', value=True)
        grid_w = CheckBox(label='xy grid', value=False)
        grid_w.changed.connect(lambda *_: setattr(grid_layer, 'visible', grid_w.value))

        colors = ['orange', 'red', 'cyan', 'yellow', 'magenta', 'lime', 'white', 'blue']
        if self.boundary_color not in colors:
            colors = [self.boundary_color] + colors
        color_w = ComboBox(label='boundary color', choices=colors, value=self.boundary_color)

        def set_bcolor(*_):
            state['brgb'] = to_rgb(color_w.value)
            cm = Colormap([[0, 0, 0], list(state['brgb'])], name='boundary')
            bound_layer.colormap = cm
            if 'preview_boundaries' in viewer.layers:
                viewer.layers['preview_boundaries'].colormap = cm

        color_w.changed.connect(set_bcolor)

        def set_rotation(*_):
            state['hist'] = make_hist(rot_w.value)
            hist_layer.data = state['hist']
            slice_pts.data = np.empty((0, 2))  # slice points are stale once the image rotates
            renumber(slice_pts)
            state['expect'] = 'atlas' if len(atlas_pts.data) == 0 else (
                'slice' if len(atlas_pts.data) > len(slice_pts.data) else 'atlas')
            status.value = f'rotated {rot_w.value}° — re-pick the slice points'

        rot_w.changed.connect(set_rotation)

        def info_text() -> str:
            return (f"index {state['index']} = {state.get('ref_mm', '?')} mm from Bregma   ·   "
                    f"dw {state['dw'] * res} µm, dh {state['dh'] * res} µm")

        info_w = Label(value=info_text())
        status = Label(value='click an atlas landmark (left), then its match on the slice (right)')
        # make the live instruction line stand out (applies to every status.value message)
        status.native.setStyleSheet('font-size: 15px; font-weight: bold; color: #ffcc00;')
        status.native.setWordWrap(True)

        def refresh(*_):
            state['index'], state['dw'], state['dh'] = idx_w.value, dw_w.value, dh_w.value
            atlas_layer.data = plane_image()
            bound_layer.data = boundary_mask(state['ann'])
            info_w.value = info_text()

        idx_w.changed.connect(refresh)
        dw_w.changed.connect(refresh)
        dh_w.changed.connect(refresh)

        @viewer.mouse_move_callbacks.append
        def on_move(_v, event):
            y, x = event.position
            viewer.text_overlay.text = region_name(y, x)

        def collect() -> tuple[np.ndarray, np.ndarray]:
            # napari points are (row, col); convert to (x, y), un-translate the slice side
            a = atlas_pts.data[:, ::-1]
            s = (slice_pts.data - np.array([0, atlas_w]))[:, ::-1]
            return s, a

        def on_preview():
            s, a = collect()
            try:
                m = estimate_transform(s, a, affine=self.affine)
            except ValueError as e:
                status.value = f'preview failed: {e}'
                return
            # inverse-warp the atlas boundaries into raw-slice space so they overlay the
            # (unmodified) histology -> points stay visible and can still be added/adjusted
            bmask = boundary_mask(state['ann'])
            hh, ww = state['hist'].shape[:2]
            binv = cv2.warpPerspective(bmask, np.linalg.inv(m), (ww, hh))
            off = (0, atlas_w)  # on the histology (right) side
            if 'preview_boundaries' in viewer.layers:
                pv = viewer.layers['preview_boundaries']
                pv.data = binv
            else:
                pv = viewer.add_image(binv, name='preview_boundaries',
                                      blending='additive', opacity=0.9, translate=off)
            pv.colormap = bound_layer.colormap  # keep preview boundaries the same color as the atlas panel
            # keep the points on top so they stay visible/clickable over the overlay
            for layer in (atlas_pts, slice_pts):
                viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)
            status.value = 'preview: atlas boundaries on your slice — keep adjusting points, then Exit preview'

        def on_exit_preview():
            if 'preview_boundaries' in viewer.layers:
                viewer.layers.remove('preview_boundaries')
            status.value = 'preview closed'

        def on_save():
            s, a = collect()
            try:
                m = estimate_transform(s, a, affine=self.affine)
            except ValueError as e:
                status.value = f'save failed: {e}'
                return
            contrast = tuple(float(v) for v in hist_layer.contrast_limits)
            save_transform(m, output_dir=out_dir, name=name, plane=self.cut_plane,
                           resolution=self.resolution, slice_index=state['index'],
                           dw=state['dw'], dh=state['dh'], slice_xy=s, atlas_xy=a,
                           rotate=rot_w.value, flip_lr=self.flip_lr, flip_ud=self.flip_ud,
                           contrast=contrast)

            # warped histology in atlas space, and a copy with the boundaries burned in.
            # bake the layer's contrast window so the .tif matches what you see.
            warped = _to_u8(apply_transformation(state['hist'], m), contrast)
            trans_path = out_dir / f'{name}_transformed.tif'
            iio.imwrite(trans_path, warped)
            print_save(trans_path)

            rgb = warped if warped.ndim == 3 else np.stack([warped] * 3, axis=-1)
            rgb = rgb[..., :3].copy()
            # use the color currently shown on the left atlas panel so the saved overlay matches
            bcol = np.asarray(bound_layer.colormap.colors[-1])[:3]
            rgb[boundary_mask(state['ann']).astype(bool)] = tuple(int(c * 255) for c in bcol)
            overlay_path = out_dir / f'{name}_overlay.tif'
            iio.imwrite(overlay_path, rgb)
            print_save(overlay_path)

            status.value = f'saved {name} transform (.npy/.json) + transformed/overlay .tif'

        def on_undo():
            # remove the most recently added point and restore the alternation state
            if len(atlas_pts.data) > len(slice_pts.data):
                atlas_pts.data = atlas_pts.data[:-1]
                renumber(atlas_pts)
                state['expect'] = 'atlas'
            elif len(slice_pts.data):
                slice_pts.data = slice_pts.data[:-1]
                renumber(slice_pts)
                state['expect'] = 'slice'
            status.value = f'{len(slice_pts.data)} complete pair(s); next: {state["expect"]} point'

        def on_clear():
            for layer in (atlas_pts, slice_pts):
                layer.data = np.empty((0, 2))
                renumber(layer)
            state['expect'] = 'atlas'
            status.value = 'cleared all points'

        preview_btn = PushButton(text='Preview overlay')
        preview_btn.changed.connect(on_preview)
        exit_preview_btn = PushButton(text='Exit preview')
        exit_preview_btn.changed.connect(on_exit_preview)
        save_btn = PushButton(text='Save transform')
        save_btn.changed.connect(on_save)
        undo_btn = PushButton(text='Undo last point')
        undo_btn.changed.connect(on_undo)
        clear_btn = PushButton(text='Clear all points')
        clear_btn.changed.connect(on_clear)

        if load.get('slice_xy') or load.get('atlas_xy'):
            ax = np.asarray(load.get('atlas_xy', []), dtype=float).reshape(-1, 2)
            sx = np.asarray(load.get('slice_xy', []), dtype=float).reshape(-1, 2)
            atlas_pts.data = ax[:, ::-1] if len(ax) else np.empty((0, 2))           # (x,y)->(row,col)
            slice_pts.data = sx[:, ::-1] + np.array([0, atlas_w]) if len(sx) else np.empty((0, 2))
            renumber(atlas_pts)
            renumber(slice_pts)
            state['expect'] = 'atlas' if len(ax) == len(sx) else 'slice'
            status.value = f'resumed: {len(sx)} pair(s) loaded'

        viewer.window.add_dock_widget(
            Container(widgets=[idx_w, dw_w, dh_w, rot_w, info_w, pick_w, grid_w, color_w,
                               undo_btn, clear_btn, preview_btn, exit_preview_btn, save_btn, status]),
            area='right', name='register'
        )
        fprint(f'registering {name}: pick points, Preview to verify, Save when done')
        napari.run()


if __name__ == '__main__':
    RegisterOptions().main()
