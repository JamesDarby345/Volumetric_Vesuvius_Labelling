# config.py

import yaml
from pathlib import Path
import os

# config.py

class CubeConfig:
    def __init__(self, cube_info):
        self.cube_info = cube_info

    def __getattr__(self, name):
        return self.cube_info.get(name)

    @property
    def zyx(self):
        return self.cube_info.get('zyx')

    @zyx.setter
    def zyx(self, value):
        if isinstance(value, str) and len(value.split('_')) == 3:
            self.cube_info['zyx'] = value
            z, y, x = value.split('_')
            self.z = z
            self.y = y
            self.x = x
        else:
            raise ValueError("ZYX must be a string in the format 'ZZZZZ_YYYYY_XXXXX'")

    @property
    def z(self):
        if self.zyx:
            return self.zyx.split('_')[0]
        return self.cube_info.get('z', '02000')

    @z.setter
    def z(self, value):
        value = str(value).zfill(5)
        self.cube_info['z'] = value
        if 'zyx' in self.cube_info:
            y, x = self.cube_info['zyx'].split('_')[1:]
            self.cube_info['zyx'] = f"{value}_{y}_{x}"

    @property
    def y(self):
        if self.zyx:
            return self.zyx.split('_')[1]
        return self.cube_info.get('y', '02000')

    @y.setter
    def y(self, value):
        value = str(value).zfill(5)
        self.cube_info['y'] = value
        if 'zyx' in self.cube_info:
            z, x = self.cube_info['zyx'].split('_')[0], self.cube_info['zyx'].split('_')[2]
            self.cube_info['zyx'] = f"{z}_{value}_{x}"

    @property
    def x(self):
        if self.zyx:
            return self.zyx.split('_')[2]
        return self.cube_info.get('x', '02000')

    @x.setter
    def x(self, value):
        value = str(value).zfill(5)
        self.cube_info['x'] = value
        if 'zyx' in self.cube_info:
            z, y = self.cube_info['zyx'].split('_')[:2]
            self.cube_info['zyx'] = f"{z}_{y}_{value}"

    def update_coordinates(self, z=None, y=None, x=None):
        if z is not None:
            self.z = z
        if y is not None:
            self.y = y
        if x is not None:
            self.x = x
        
        if z is not None and y is not None and x is not None:
            self.zyx = f"{self.z}_{self.y}_{self.x}"
        elif 'zyx' in self.cube_info:
            del self.cube_info['zyx']

    @property
    def z_num(self):
        return int(self.z)

    @property
    def y_num(self):
        return int(self.y)

    @property
    def x_num(self):
        return int(self.x)
    
    @property
    def cc_min_size(self):
        return self.cube_info.get('cc_min_size', 800)
    
    @property
    def align_coordinates(self):
        return self.cube_info.get('align_coordinates', True)
    
    @property
    def smoother_labels(self):
        return self.cube_info.get('smoother_labels', False)
    
    @property
    def create_papyrus_mask_if_not_provided(self):
        return self.cube_info.get('create_papyrus_mask_if_not_provided', True)

    @property
    def scroll_name(self):
        return self.cube_info.get('scroll_name', "")

    @property
    def chunk_size(self):
        return self.cube_info.get('chunk_size', 256)
    
    @chunk_size.setter
    def chunk_size(self, value):
        self.cube_info['chunk_size'] = value

    @property
    def edit_chunk_size(self):
        return self.cube_info.get('edit_chunk_size', 256)

    @property
    def pad_amount(self):
        return self.cube_info.get('pad_amount', 100)
    
    @property
    def voxelized_mesh_pad_amount(self):
        return self.cube_info.get('voxelized_mesh_pad_amount', 20)

    @property
    def brush_size(self):
        return self.cube_info.get('brush_size', 4)

    @property
    def main_label_layer_name(self):
        return self.cube_info.get('main_label_layer_name', 'papyrus')

    @property
    def nrrd_cube_path(self):
        nrrd_cube_path = self.cube_info.get('nrrd_cube_path', '')
        if nrrd_cube_path is None or nrrd_cube_path == '':
            nrrd_cube_path = os.path.join(os.getcwd(), 'data', 'nrrd_cubes', self.cube_info.get('scroll_name', '')) 
        return nrrd_cube_path

    @property
    def raw_data_zarr_path(self):
        return self.cube_info.get('raw_data_zarr_path', '')

    @property
    def ink_pred_zarr_path(self):
        return self.cube_info.get('ink_pred_zarr_path', '')

    @property
    def raw_data_axis_order(self):
        return self.cube_info.get('raw_data_axis_order', 'zyx')

    @property
    def ink_pred_label_order(self):
        return self.cube_info.get('ink_pred_label_order', 'zyx')

    @property
    def ink_threshold(self):
        return self.cube_info.get('ink_threshold', 150)

    @property
    def author(self):
        return self.cube_info.get('author', '')

    @property
    def factor(self):
        return self.cube_info.get('factor', 1.0)

    @property
    def using_raw_data_zarr(self):
        return bool(self.raw_data_zarr_path)

    @property
    def using_ink_pred_zarr(self):
        return bool(self.ink_pred_zarr_path)

class HotkeyConfig:
    def __init__(self, hotkey_config):
        self.hotkey_config = hotkey_config

    def __getattr__(self, name):
        return self.hotkey_config.get(name)

class Config:
    def __init__(self, config_path='napari_config.yaml'):
        self.config_path = Path(config_path)
        self.cube_config = None
        self.hotkey_config = None
        self.load_config()

    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.cube_config = CubeConfig(config.get('cube_info', {}))
                self.hotkey_config = HotkeyConfig(config.get('customizable_hotkeys', {}))
        else:
            print(f"Config file not found: {self.config_path}")