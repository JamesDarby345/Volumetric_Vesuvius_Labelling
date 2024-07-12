# config.py

import yaml
from pathlib import Path

class CubeConfig:
    def __init__(self, cube_info):
        self.cube_info = cube_info

    def __getattr__(self, name):
        return self.cube_info.get(name)

    @property
    def zyx(self):
        return self.cube_info.get('zyx')

    @property
    def z(self):
        if self.zyx:
            return self.zyx.split('_')[0]
        return self.cube_info.get('z', '02000')

    @property
    def y(self):
        if self.zyx:
            return self.zyx.split('_')[1]
        return self.cube_info.get('y', '02000')

    @property
    def x(self):
        if self.zyx:
            return self.zyx.split('_')[2]
        return self.cube_info.get('x', '02000')

    @property
    def z_num(self):
        return int(self.z)

    @property
    def y_num(self):
        return int(self.y)

    @property
    def x_num(self):
        return int(self.x)
    
    def update_coordinates(self, z=None, y=None, x=None):
        if z is not None:
            self.cube_info['z'] = str(z).zfill(5)
        if y is not None:
            self.cube_info['y'] = str(y).zfill(5)
        if x is not None:
            self.cube_info['x'] = str(x).zfill(5)
        
        if z is not None and y is not None and x is not None:
            self.cube_info['zyx'] = f"{self.z}_{self.y}_{self.x}"
        elif 'zyx' in self.cube_info:
            del self.cube_info['zyx']

    @property
    def scroll_name(self):
        return self.cube_info.get('scroll_name', "")

    @property
    def chunk_size(self):
        return self.cube_info.get('chunk_size', 256)

    @property
    def pad_amount(self):
        return self.cube_info.get('pad_amount', 100)

    @property
    def brush_size(self):
        return self.cube_info.get('brush_size', 4)

    @property
    def main_label_layer_name(self):
        return self.cube_info.get('main_label_layer_name', 'papyrus')

    @property
    def nrrd_cube_path(self):
        return self.cube_info.get('nrrd_cube_path', '')

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