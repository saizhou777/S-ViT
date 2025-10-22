import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UTB180Dataset(BaseDataset):
    """
    UTB180 dataset.
    目录结构:
    <utb180_root>/
    Video_0001/
    imgs/0001.jpg ...
    groundtruth_rect.txt
    ...
    Video_0180/
    UTB.json  (可选，包含各序列的 img_names 与 gt_rect)
    """
    def __init__(self, use_json=False):
        super().__init__()
        # 需要在 local_env_settings() 里设置 settings.utb180_path
        self.base_path = getattr(self.env_settings, 'utb180_path', None)
        if self.base_path is None or not os.path.isdir(self.base_path):
            raise RuntimeError('utb180_path 未在 local.py 中配置或目录不存在')


        self.use_json = use_json
        self.json_path = os.path.join(self.base_path, 'UTB.json')
        self.sequence_list = self._get_sequence_list()

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _get_sequence_list(self):
        vids = [d for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d)) and d.startswith('Video_')]
        vids.sort()
        if len(vids) == 0 and os.path.isfile(self.json_path):
            with open(self.json_path, 'r') as f:
                j = json.load(f)
            vids = sorted(list(j.keys()))
        return vids

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)
        imgs_dir = os.path.join(seq_dir, 'imgs')
        gt_path = os.path.join(seq_dir, 'groundtruth_rect.txt')

        frames = None
        gt = None

        if not self.use_json:
            # 优先使用磁盘中的 imgs + groundtruth_rect.txt
            if os.path.isdir(imgs_dir):
                frame_list = [f for f in os.listdir(imgs_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                # 数字优先排序（兼容 0001.jpg）
                def key_fn(fn):
                    base = os.path.splitext(fn)[0]
                    try:
                        return int(base)
                    except:
                        return base
                frame_list.sort(key=key_fn)
                frames = [os.path.join(imgs_dir, f) for f in frame_list]

            if os.path.isfile(gt_path):
                # 兼容逗号或空格分隔
                gt = load_text(str(gt_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
                if gt.ndim == 1:
                    gt = gt.reshape(1, -1)

        # 如有必要，回退到 UTB.json
        if (frames is None or len(frames) == 0 or gt is None) and os.path.isfile(self.json_path):
            with open(self.json_path, 'r') as f:
                J = json.load(f)
            meta = J[sequence_name]
            frames = [os.path.join(self.base_path, p) for p in meta['img_names']]
            gt = np.array(meta['gt_rect'], dtype=np.float64)

        assert frames is not None and len(frames) > 0, f'找不到图像帧: {sequence_name}'
        assert gt is not None and len(gt) == len(frames), f'标注与帧数不一致: {sequence_name}, gt={len(gt)}, frames={len(frames)}'

        return Sequence(sequence_name, frames, 'utb180', gt.reshape(-1, 4))