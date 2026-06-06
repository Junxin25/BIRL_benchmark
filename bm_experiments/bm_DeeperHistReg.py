"""
Benchmark wrapper for DeeperHistReg.

This adapter runs DeeperHistReg as an external command and converts its saved
displacement field into BIRL landmark output. It does not modify the
DeeperHistReg or SuperPoint repositories.

Example
-------
Run a smoke benchmark on BIRL sample pairs::

    python bm_experiments/bm_DeeperHistReg.py \
        -t ./data-images/pairs-imgs-lnds_mix.csv \
        -o ./results \
        -py /home/junxinfu/.conda/envs/Deeperhistreg/bin/python \
        -script /home/junxinfu/registration/Model/DeeperHistReg/deeperhistreg/run.py \
        -params /home/junxinfu/registration/Model/DeeperHistReg/deeperhistreg_params/default_initial_swint_superglue_optimized.json \
        --visual --unique --nb_workers 1
"""

import logging
import os
import re
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.data_io import load_landmarks, save_landmarks
from bm_experiments import bm_comp_perform


class BmDeeperHistReg(ImRegBenchmark):
    """BIRL benchmark adapter for DeeperHistReg."""

    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS + [
        'exec_Python',
        'path_script',
        'path_params',
    ]

    NAME_DISPLACEMENT_FIELD = 'displacement_field.mha'
    NAME_POSTPROCESSING_PARAMS = 'postprocessing_params.json'
    NAME_DHR_LOG = 'logs.txt'
    NAME_TEMP_DIR = 'temp'
    NAME_LNDS_WARPED = 'warped-target-landmarks.csv'
    NAME_WARPED_SOURCE_PATTERN = 'warped_source'
    DEFAULT_CASE_NAME = 'BIRL_DeeperHistReg'

    def _prepare(self):
        """Copy the DeeperHistReg JSON config into the experiment folder."""
        logging.info('-> copy DeeperHistReg configuration...')
        self._copy_config_to_expt('path_params')

    def _generate_regist_command(self, item):
        """Generate the DeeperHistReg command for one registration pair."""
        path_dir = self._get_path_reg_dir(item)
        path_im_ref, path_im_move, _, _ = self._get_paths(item)
        temp_dir = os.path.join(path_dir, self.NAME_TEMP_DIR)
        case_name = '%s_%s' % (self.params.get('case_name', self.DEFAULT_CASE_NAME), item['ID'])

        cmd = [
            self.params['exec_Python'],
            self.params['path_script'],
            '--srcp', path_im_move,
            '--trgp', path_im_ref,
            '--out', path_dir,
            '--params', self.params['path_params'],
            '--exp', case_name,
            '--temp', temp_dir,
            '--sdf',
        ]
        if self.params.get('copy_target', False):
            cmd.append('--cpt')
        if self.params.get('delete_temporary_results', False):
            cmd.append('--dtmp')
        return ' '.join(cmd)

    def _extract_warped_image_landmarks(self, item):
        """Convert DeeperHistReg outputs into BIRL result paths."""
        path_dir = self._get_path_reg_dir(item)
        path_displacement = os.path.join(path_dir, self.NAME_DISPLACEMENT_FIELD)
        if not os.path.isfile(path_displacement):
            logging.warning('Missing DeeperHistReg displacement field: %s', path_displacement)
            return {}

        path_img_warp = self._find_warped_source(path_dir)
        path_lnds_ref = self._get_paths(item)[2]
        path_lnds_warp = os.path.join(path_dir, self.NAME_LNDS_WARPED)
        landmarks_ref = load_landmarks(path_lnds_ref)
        displacement = self._load_displacement_field(path_displacement)
        warped_landmarks = self._warp_landmarks_with_backward_field(landmarks_ref, displacement)
        save_landmarks(path_lnds_warp, warped_landmarks)

        return {
            self.COL_IMAGE_MOVE_WARP: path_img_warp,
            self.COL_POINTS_REF_WARP: path_lnds_warp,
        }

    def _extract_execution_time(self, item):
        """Read DeeperHistReg total time from copied logs if available."""
        path_dir = self._get_path_reg_dir(item)
        path_log = os.path.join(path_dir, self.NAME_DHR_LOG)
        if not os.path.isfile(path_log):
            return None
        with open(path_log, 'r') as fp:
            text = fp.read()
        matches = re.findall(r'Total registration time:\s*([0-9.]+)\s*seconds', text)
        if not matches:
            return None
        return float(matches[-1]) / 60.

    @staticmethod
    def _find_warped_source(path_dir):
        for name in sorted(os.listdir(path_dir)):
            if BmDeeperHistReg.NAME_WARPED_SOURCE_PATTERN in name:
                return os.path.join(path_dir, name)
        logging.warning('Missing DeeperHistReg warped source image in: %s', path_dir)
        return None

    @staticmethod
    def _load_displacement_field(path_displacement):
        displacement = sitk.GetArrayFromImage(sitk.ReadImage(path_displacement)).astype(np.float32)
        if displacement.ndim != 3:
            raise ValueError('Expected displacement field with shape (2, H, W), got %r' % (displacement.shape,))
        if displacement.shape[0] != 2 and displacement.shape[-1] == 2:
            displacement = np.moveaxis(displacement, -1, 0)
        if displacement.shape[0] != 2:
            raise ValueError('Expected two displacement channels, got %r' % (displacement.shape,))
        return displacement

    @staticmethod
    def _warp_landmarks_with_backward_field(landmarks, displacement):
        """Map target-frame landmarks into source coordinates.

        DeeperHistReg saves the sampling field used to render warped_source in
        target space. For target coordinates (x, y), adding the displacement
        gives the corresponding source coordinates.
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        ux = ndimage.map_coordinates(displacement[0], [ys, xs], order=1, mode='nearest')
        uy = ndimage.map_coordinates(displacement[1], [ys, xs], order=1, mode='nearest')
        return np.stack((xs + ux, ys + uy), axis=1)

    @staticmethod
    def extend_parse(arg_parser):
        arg_parser.add_argument(
            '-py',
            '--exec_Python',
            type=str,
            required=True,
            help='path to the Python executable with DeeperHistReg dependencies',
        )
        arg_parser.add_argument(
            '-script',
            '--path_script',
            required=True,
            type=str,
            help='path to DeeperHistReg run.py',
        )
        arg_parser.add_argument(
            '-params',
            '--path_params',
            required=True,
            type=str,
            help='path to the DeeperHistReg JSON parameter file',
        )
        arg_parser.add_argument(
            '--case_name',
            type=str,
            required=False,
            default=BmDeeperHistReg.DEFAULT_CASE_NAME,
            help='case-name prefix passed to DeeperHistReg',
        )
        arg_parser.add_argument(
            '--copy_target',
            action='store_true',
            required=False,
            default=False,
            help='ask DeeperHistReg to copy target images into each output folder',
        )
        arg_parser.add_argument(
            '--dtmp',
            dest='delete_temporary_results',
            action='store_true',
            required=False,
            default=False,
            help='ask DeeperHistReg to delete temporary results after each case',
        )
        return arg_parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmDeeperHistReg.main()

    if arg_params.get('run_comp_benchmark', False):
        bm_comp_perform.main(path_expt)
