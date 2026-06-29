from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from argclz import AbstractParser, argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.ccf.matrix import slice_transform_helper
from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA
from neuralib.atlas.view import get_slice_view
from neuralib.util.verbose import fprint, print_save

from ccf2d.core import boundary_mask, plane_point_to_ccf_mm, read_oriented, rotate

__all__ = ['ProbeOptions']

# superficial = dorsal (top of brain, small DV); deep = ventral. ProbeRenderCLI wants dorsal first.
MARKS = ('superficial', 'deep')
_POINT = {'superficial': 'dorsal', 'deep': 'ventral'}


class ProbeOptions(AbstractParser):
    DESCRIPTION = ('Reconstruct probe shanks from dye labels on registered slices (napari). '
                   'Pick superficial+deep per shank across serial sections, then render with brainrender.')

    directory: Path | None = argument(
        '-D', '--directory',
        default=None,
        help='folder of serial sections (steps through them; reads <dir>/transformations/<stem>_transform.json)'
    )

    raw_image: Path | None = argument(
        '-I', '--image',
        default=None,
        help='single registered histology image (alternative to -D)'
    )

    transform_dir: Path | None = argument(
        '--transform-dir',
        default=None,
        help='where the *_transform.json live (default: <image-dir>/transformations)'
    )

    output: Path | None = argument(
        '-O', '--output',
        default=None,
        help='output points csv (default: <dir>/probe_shanks.csv)'
    )

    depth: int | None = argument(
        '--depth',
        default=None,
        help='implant depth (µm); if set, render adds the theoretical track, else dye-only'
    )

    interval: int | None = argument(
        '--interval',
        default=None,
        help='shank interval (µm) for a multi-shank theoretical track'
    )

    _IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    def _list_images(self, d: Path) -> list[Path]:
        return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in self._IMG_EXT)

    def run(self):
        files = self._list_images(self.directory) if self.directory else []
        if files and self.raw_image is None:
            self.raw_image = files[0]
        if self.raw_image is None:
            raise ValueError('provide a registered image via -I/--image or a folder via -D/--directory')

        base = self.raw_image.parent
        self._tdir = self.transform_dir or base / 'transformations'
        self._out = self.output or base / 'probe_shanks.csv'

        first = self._transform_path(self.raw_image)
        if not first.exists():
            raise FileNotFoundError(
                f'no native registration {first} — register with `ccf2d register` first '
                f'(legacy MATLAB .mat is not supported)')
        self._launch_napari(files)

    def _transform_path(self, img: Path) -> Path:
        return self._tdir / f'{img.stem}_transform.json'

    def _launch_napari(self, files: list[Path]):
        import napari
        from magicgui.widgets import ComboBox, Container, Label, PushButton, SpinBox

        viewer = napari.Viewer(title='ccf2d probe')
        viewer.text_overlay.visible = True

        # pts[(shank, 'dorsal'|'ventral')] = (AP, DV, ML) in bregma-relative mm
        state: dict = {'pts': {}, 'files': files, 'cursor': 0, 'plane': None, 'view': None,
                       'plane_off': None, 'name': None}

        warped_layer = None  # created on first load (re-created to switch grayscale<->RGB safely)
        bound_layer = viewer.add_image(np.zeros((10, 10)), name='boundaries',
                                       colormap='red', blending='additive', opacity=0.7)
        click_layer = viewer.add_points(name='clicks', face_color='cyan', symbol='cross',
                                        size=20, ndim=2)

        def set_histology(img: np.ndarray):
            # napari can't flip grayscale<->RGB in place; re-create and keep it at the bottom
            nonlocal warped_layer
            if warped_layer is not None and warped_layer in viewer.layers:
                viewer.layers.remove(warped_layer)
            warped_layer = viewer.add_image(img, name='histology', colormap='gray')
            viewer.layers.move(viewer.layers.index(warped_layer), 0)

        status = Label(value='load a registered slice to begin')
        status.native.setStyleSheet('font-size: 14px; font-weight: bold; color: #ffcc00;')
        status.native.setWordWrap(True)
        summary = Label(value='')
        summary.native.setWordWrap(True)

        def load_slice(img: Path):
            tp = self._transform_path(img)
            if not tp.exists():
                status.value = f'no registration for {img.name} — skip or register it first'
                return

            meta = json.loads(tp.read_text())
            plane, res = meta['plane'], int(meta['resolution'])
            idx, dw, dh = int(meta['slice_index']), int(meta['dw']), int(meta['dh'])
            oimg = rotate(read_oriented(img, meta.get('flip_lr', False), meta.get('flip_ud', False)),
                          float(meta.get('rotate', 0.0)))
            _, trans = slice_transform_helper(oimg, np.array(meta['matrix'], float), plane_type=plane)

            od = lambda v: v + 1 if v != 0 else 0
            view = get_slice_view('reference', plane, resolution=res)
            ann = get_slice_view('annotation', plane, resolution=res)
            sp = view.plane_at(idx).with_offset(od(dw), od(dh))
            ann_sp = ann.plane_at(idx).with_offset(od(dw), od(dh))

            state.update(plane=plane, view=view, plane_off=sp.plane_offset, name=img.stem)
            set_histology(trans)
            bound_layer.data = boundary_mask(ann_sp.image)
            click_layer.data = np.empty((0, 2))
            status.value = f'{img.name}: click the dye, marking {mark_w.value} of shank {shank_w.value}'

        def refresh_summary():
            if not state['pts']:
                summary.value = 'no points yet'
                return
            rows = []
            for shank in sorted({s for s, _ in state['pts']}):
                have = [p for (s, p) in state['pts'] if s == shank]
                rows.append(f'shank {shank}: ' + ', '.join(sorted(have)))
            summary.value = '\n'.join(rows)

        @viewer.mouse_drag_callbacks.append
        def on_click(_v, event):
            dragged = False
            yield
            while event.type == 'mouse_move':
                dragged = True
                yield
            if dragged or state['view'] is None:
                return
            y, x = event.position
            off = state['plane_off']
            if not (0 <= y < off.shape[0] and 0 <= x < off.shape[1]):
                return
            ccf = plane_point_to_ccf_mm(off[int(y), int(x)], x, y,
                                        project_index=state['view'].project_index,
                                        resolution=state['view'].resolution,
                                        bregma_10um=tuple(ALLEN_CCF_10um_BREGMA))
            key = (int(shank_w.value), _POINT[mark_w.value])
            state['pts'][key] = ccf
            click_layer.add(np.array([[y, x]]))
            status.value = (f'shank {shank_w.value} {mark_w.value} -> '
                            f'AP {ccf[0]:.2f}, DV {ccf[1]:.2f}, ML {ccf[2]:.2f} mm')
            refresh_summary()

        shank_w = SpinBox(label='shank', value=1, min=1, max=64)
        mark_w = ComboBox(label='marking', choices=list(MARKS), value=MARKS[0])

        def save_csv() -> Path | None:
            shanks = sorted({s for s, _ in state['pts']})
            rows = []
            for s in shanks:
                for point in ('dorsal', 'ventral'):  # dorsal first: ProbeRenderCLI reshape order
                    if (s, point) not in state['pts']:
                        status.value = f'shank {s} missing its {point} point'
                        return None
                    ap, dv, ml = state['pts'][(s, point)]
                    rows.append({'AP_location': ap, 'DV_location': dv, 'ML_location': ml,
                                 'probe_idx': s, 'point': point})
            if not rows:
                status.value = 'no points to save'
                return None
            self._out.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows).write_csv(self._out)
            print_save(self._out)
            status.value = f'saved {len(shanks)} shank(s) -> {self._out.name}'
            return self._out

        def on_render():
            csv = save_csv()
            if csv is None:
                return
            plane = state['plane'] or 'coronal'
            cmd = [sys.executable, '-m', 'neuralib.atlas.brainrender.probe',
                   '--file', str(csv), '--plane-type', plane]
            if self.depth is None:
                cmd.append('--dye')
            else:
                cmd += ['--depth', str(self.depth)]
                if self.interval is not None:
                    cmd += ['--interval', str(self.interval)]
            status.value = 'rendering with brainrender (separate window)...'
            subprocess.Popen(cmd)

        def on_undo_shank():
            for point in ('ventral', 'dorsal'):
                if (int(shank_w.value), point) in state['pts']:
                    del state['pts'][(int(shank_w.value), point)]
                    status.value = f'removed shank {shank_w.value} {point}'
                    break
            refresh_summary()

        def step(delta: int):
            files = state['files']
            if not files:
                status.value = 'single-image mode — nothing to step'
                return
            state['cursor'] = int(np.clip(state['cursor'] + delta, 0, len(files) - 1))
            load_slice(files[state['cursor']])

        prev_btn = PushButton(text='◀ Prev slice')
        prev_btn.changed.connect(lambda *_: step(-1))
        next_btn = PushButton(text='Next slice ▶')
        next_btn.changed.connect(lambda *_: step(+1))
        undo_btn = PushButton(text='Undo this shank')
        undo_btn.changed.connect(lambda *_: on_undo_shank())
        save_btn = PushButton(text='Save CSV')
        save_btn.changed.connect(lambda *_: save_csv())
        render_btn = PushButton(text='Render (brainrender)')
        render_btn.changed.connect(lambda *_: on_render())

        def header(text):
            lbl = Label(value=text)
            lbl.native.setStyleSheet('font-weight: bold; color: #88c0d0; padding-top: 6px;')
            return lbl

        def srow(*ws):
            return Container(widgets=list(ws), layout='horizontal', labels=False)

        panel = Container(widgets=[
            header('Slice'), srow(prev_btn, next_btn),
            header('Dye point'), shank_w, mark_w, undo_btn,
            header('Accumulated'), summary,
            header('Output'), srow(save_btn, render_btn),
            status,
        ])
        viewer.window.add_dock_widget(panel, area='right', name='probe')

        # touch the atlas once so structures are cached before the first click feels snappy
        BrainGlobeAtlas('allen_mouse_10um')
        if state['files']:
            load_slice(state['files'][0])
        elif self.raw_image:
            load_slice(self.raw_image)
        refresh_summary()

        fprint('probe: pick superficial+deep dye per shank (step slices as needed), then Render')
        napari.run()


if __name__ == '__main__':
    ProbeOptions().main()
