from pathlib import Path

from poregen.models.get_model import build_ddconfig
from poregen.trainers.trainers import (
    _experimental_output_folder,
    _repo_root_from_cfg,
)


def test_build_ddconfig_geomodel_vs_bps_defaults():
    geomodel = build_ddconfig({
        'resolution': 128,
        'has_mid_attn': True,
        'ch': 64,
        'ch_mult': [1, 2, 4, 8],
        'z_channels': 1,
        'num_res_blocks': 1,
        'dropout': 0.1,
        'double_z': True,
        'in_channels': 1,
        'out_ch': 1,
    })
    assert geomodel.ch == 64
    assert list(geomodel.ch_mult) == [1, 2, 4, 8]
    assert geomodel.z_channels == 1
    assert geomodel.num_res_blocks == 1
    assert geomodel.has_mid_attn is True
    assert geomodel.resolution == 128

    nested = build_ddconfig({
        'config': {
            'resolution': 128,
            'has_mid_attn': True,
            'ch': 64,
            'ch_mult': [1, 2, 4, 8],
            'z_channels': 1,
        }
    })
    assert nested.ch == 64
    assert nested.z_channels == 1
    assert nested.has_mid_attn is True


def test_nested_geomodel_cfg_writes_under_savedmodels():
    repo = _repo_root_from_cfg(
        Path('/home/lcadame/repos/PoreGen/configs/bps/geomodels/20260901-ldm-federico.yaml')
    )
    assert repo.name == 'PoreGen'
    folder = _experimental_output_folder(
        {'output': {}},
        Path('/home/lcadame/repos/PoreGen/configs/bps/geomodels/20260901-ldm-federico.yaml'),
    )
    assert folder == repo / 'savedmodels' / 'experimental' / '20260901-ldm-federico'

    explicit = _experimental_output_folder(
        {'output': {'folder': '/tmp/custom-ldm'}},
        Path('/home/lcadame/repos/PoreGen/configs/bps/geomodels/x.yaml'),
    )
    assert explicit == Path('/tmp/custom-ldm')


if __name__ == '__main__':
    test_build_ddconfig_geomodel_vs_bps_defaults()
    test_nested_geomodel_cfg_writes_under_savedmodels()
    print('All tests passed')
