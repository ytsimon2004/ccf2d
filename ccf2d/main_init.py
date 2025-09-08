import shutil
from pathlib import Path

import numpy as np
from argclz import AbstractParser, argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.data import ATLAS_NAME
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.unstable import unstable
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import fprint, print_save

from ccf2d.io import PROJECT_DIRECTORY

__all__ = ['InitOptions']


class InitOptions(AbstractParser):
    DESCRIPTION = 'Options for initializing project (downloading reference files)'

    atlas_name: ATLAS_NAME = argument(
        '--atlas',
        default='allen_mouse_10um',
        help='brain atlas name'
    )

    force_download: bool = argument(
        '--force',
        help='force re-download'
    )

    use_brainglobe_source: bool = argument(
        '--use-brainglobe-source',
        help='use brainglobe atlas source instead of allenccf'
    )

    def run(self):
        fprint(f'Initialing project (relevant cache downloading (~5GB) ...)')
        if self.use_brainglobe_source:
            self.brainglobe_init()
        else:
            self.allenccf_init()

    def allenccf_init(self):
        """
        .. seealso:: `<https://github.com/cortex-lab/allenCCF>`_
        """
        root_dir = ensure_dir(PROJECT_DIRECTORY)

        # annotation
        annotation_url = 'https://figshare.com/ndownloader/files/44925493'
        output = root_dir / 'annotation_volume_10um_by_index.npy'
        if not output.exists() or self.force_download:
            av = download_with_tqdm(annotation_url)
            with open(output, 'wb') as f:
                f.write(av.getvalue())
                print_save(output, verb='DOWNLOAD')

        # template
        template_url = 'https://figshare.com/ndownloader/files/44925496'
        output = root_dir / 'template_volume_10um.npy'
        if not output.exists() or self.force_download:
            tv = download_with_tqdm(template_url)
            with open(output, 'wb') as f:
                f.write(tv.getvalue())
                print_save(output, verb='DOWNLOAD')

        # structures
        src = Path(__file__).parents[1] / 'res' / 'allenccf' / 'structure_tree_safe_2017.csv'
        dst = root_dir / 'structure_tree_safe_2017.csv'
        shutil.copy(src, dst)
        print_save(dst, verb='COPY')

    @unstable(method='brainglobe_init')
    def brainglobe_init(self):
        """TODO not test yet"""
        bg = BrainGlobeAtlas(self.atlas_name, check_latest=True)
        np.save(bg.root_dir / 'reference.npy', bg.reference)
        np.save(bg.root_dir / 'annotation.npy', bg.annotation)


if __name__ == '__main__':
    InitOptions().main()
