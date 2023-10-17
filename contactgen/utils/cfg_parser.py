import os
import yaml


class Config(dict):

    def __init__(self,default_cfg_path=None,**kwargs):

        default_cfg = {}
        if default_cfg_path is not None and os.path.exists(default_cfg_path):
            default_cfg = self.load_cfg(default_cfg_path)

        super(Config,self).__init__(**kwargs)

        default_cfg.update(self)
        self.update(default_cfg)
        self.default_cfg = default_cfg

    def load_cfg(self,load_path):
        with open(load_path, 'r') as infile:
            cfg = yaml.safe_load(infile)
        return cfg if cfg is not None else {}

    def write_cfg(self,write_path=None):

        if write_path is None:
            write_path = 'yaml_config.yaml'

        dump_dict = {k:v for k,v in self.items() if k!='default_cfg'}
        with open(write_path, 'w') as outfile:
            yaml.safe_dump(dump_dict, outfile, default_flow_style=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__