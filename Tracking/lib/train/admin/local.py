class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data1/saizhou777/ostrack20251012'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data1/saizhou777/ostrack20251012/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data1/saizhou777/ostrack20251012/pretrained_networks'
        self.lasot_dir = '/data2/xd/dataset/LaSOT/raw/LaSOT'
        self.got10k_dir = '/data2/xd/dataset/GOT-10k/raw/GOT-10k/train'
        self.got10k_val_dir = '/data2/xd/dataset/GOT-10k/raw/GOT-10k/val'
        self.lasot_lmdb_dir = '/data1/saizhou777/ostrack20251012/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data1/saizhou777/ostrack20251012/data/got10k_lmdb'
        self.trackingnet_dir = '/data2/xd/dataset/TrackingNet/raw/TrackingNet'
        self.trackingnet_lmdb_dir = '/data1/saizhou777/ostrack20251012/data/trackingnet_lmdb'
        self.coco_dir = '/data2/xd/dataset/coco'
        self.coco_lmdb_dir = '/data1/saizhou777/ostrack20251012/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data1/saizhou777/ostrack20251012/data/vid'
        self.imagenet_lmdb_dir = '/data1/saizhou777/ostrack20251012/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
