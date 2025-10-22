import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class UAVTrack112Dataset(BaseDataset):
    """
    UAVTrack112 dataset.

    复制
    目录结构示例:
    <uavtrack112_root>/
    anno/
        SeqA.txt
        SeqB.txt
        ...
    data_seq/
        SeqA/
        0001.jpg 0002.jpg ...
        SeqB/
        0001.jpg 0002.jpg ...
        ...

    说明:
    - anno 中的 txt 与 data_seq 下同名文件夹一一对应
    - 标注每行 4 值: x,y,w,h，分隔符为逗号或空白，自动兼容
    """

    def __init__(self):
        super().__init__()
        # 需要在 local.py 里配置 settings.uavtrack112_path
        self.base_path = getattr(self.env_settings, 'uavtrack112_path', None)
        if self.base_path is None or not os.path.isdir(self.base_path):
            raise RuntimeError('uavtrack112_path 未在 local.py 中配置或目录不存在')

        self.anno_dir = os.path.join(self.base_path, 'anno')
        self.data_dir = os.path.join(self.base_path, 'data_seq')

        if not os.path.isdir(self.anno_dir):
            raise RuntimeError(f'未找到标注目录: {self.anno_dir}')
        if not os.path.isdir(self.data_dir):
            raise RuntimeError(f'未找到图像目录: {self.data_dir}')

        self.sequence_list = self._get_sequence_list()

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _get_sequence_list(self):
        # 基于 anno 下的 .txt 构造序列名列表，并要求 data_seq 下存在同名目录
        txts = [f for f in os.listdir(self.anno_dir) if f.lower().endswith('.txt')]
        seq_names = []
        for t in txts:
            name = os.path.splitext(t)[0]
            img_dir = os.path.join(self.data_dir, name)
            if os.path.isdir(img_dir):
                seq_names.append(name)
        seq_names.sort()
        return seq_names

    def _construct_sequence(self, sequence_name):
        img_dir = os.path.join(self.data_dir, sequence_name)
        anno_path = os.path.join(self.anno_dir, sequence_name + '.txt')

        if not os.path.isfile(anno_path):
            raise RuntimeError(f'找不到标注文件: {anno_path}')
        # 读取标注（兼容逗号或空白）
        gt = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

        # 收集帧（支持 jpg/jpeg/png）
        frames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(frames) == 0:
            raise RuntimeError(f'找不到图像帧: {sequence_name} 于 {img_dir}')

        # 数字优先排序（支持 0001.jpg 或 12.png）
        def key_fn(fn):
            base = os.path.splitext(fn)[0]
            try:
                return int(base)
            except:
                return base
        frames.sort(key=key_fn)
        frames = [os.path.join(img_dir, f) for f in frames]

        # 长度对齐检查（若不一致，你也可以选择截断到最小长度）
        if len(gt) != len(frames):
            raise RuntimeError(f'标注与帧数不一致: {sequence_name}, gt={len(gt)}, frames={len(frames)}')

        return Sequence(sequence_name, frames, 'uavtrack112', gt.reshape(-1, 4))