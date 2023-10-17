import argparse
import os
from contactgen.utils.cfg_parser import Config
from contactgen.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Training')

    parser.add_argument('--work-dir', default='./exp', type=str, help='exp dir')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Training batch size')
    parser.add_argument('--lr', default=8e-4, type=float,
                        help='Training learning rate')
    parser.add_argument("--config_path", type=str, default="contactgen/configs/default.yaml")
    
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    cwd = os.getcwd()
    cfg_path = args.config_path

    cfg = {
        'batch_size': args.batch_size,
        'base_lr': args.lr,
        'base_dir': cwd,
        'work_dir': args.work_dir,
        'checkpoint': None, 
    }

    cfg = Config(default_cfg_path=cfg_path, **cfg)
    cf_trainer = Trainer(cfg=cfg)
    cf_trainer.fit()