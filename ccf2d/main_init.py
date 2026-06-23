from argclz import AbstractParser, argument
from brainglobe_atlasapi import BrainGlobeAtlas
from neuralib.atlas.data import ATLAS_NAME
from neuralib.util.verbose import fprint

__all__ = ['InitOptions']


class InitOptions(AbstractParser):
    DESCRIPTION = 'Download the brain atlas into ~/.brainglobe'

    atlas_name: ATLAS_NAME = argument(
        '--atlas',
        default='allen_mouse_10um',
        help='brainglobe atlas name'
    )

    force_download: bool = argument(
        '--force',
        help='check for and fetch the latest atlas version'
    )

    def run(self):
        fprint(f'ensuring atlas {self.atlas_name!r} in ~/.brainglobe (first download ~2GB) ...')
        bg = BrainGlobeAtlas(self.atlas_name, check_latest=self.force_download)
        fprint(f'atlas ready at {bg.root_dir}')


if __name__ == '__main__':
    InitOptions().main()
