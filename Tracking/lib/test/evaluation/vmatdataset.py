import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class VMATDataset(BaseDataset):
    """
    VMAT dataset.
    目录结构示例:
    <vmat_root>/
    annotations/
    SeqA.txt
    SeqB.txt
    ...
    SeqA/
    0001.jpg 0002.jpg ...
    SeqB/
    0001.jpg 0002.jpg ...
    ...
    注：annotations 中每个 txt 与同名序列文件夹一一对应；txt 每行 4 值 x,y,w,h（逗号或空白分隔）。
    """

    def __init__(self):
        super().__init__()
    # 需要在 local.py 里设置 settings.vmat_path
        self.base_path = getattr(self.env_settings, 'vmat_path', None)
        if self.base_path is None or not os.path.isdir(self.base_path):
            raise RuntimeError('vmat_path 未在 local.py 中配置或目录不存在')

        self.anno_dir = os.path.join(self.base_path, 'annotations')
        if not os.path.isdir(self.anno_dir):
            raise RuntimeError(f'未找到标注目录: {self.anno_dir}')

        self.sequence_list = self._get_sequence_list()

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _get_sequence_list(self):
        # 根据 annotations 下的 .txt 作为基准，取其同名文件夹作为序列
        txts = [f for f in os.listdir(self.anno_dir) if f.lower().endswith('.txt')]
        seq_names = []
        for t in txts:
            name = os.path.splitext(t)[0]
            seq_dir = os.path.join(self.base_path, name)
            if os.path.isdir(seq_dir):
                seq_names.append(name)
        seq_names.sort()
        return seq_names

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)
        anno_path = os.path.join(self.anno_dir, sequence_name + '.txt')

        # 收集帧
        frame_list = [f for f in os.listdir(seq_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 数字优先排序（兼容 0001.jpg）
        def key_fn(fn):
            base = os.path.splitext(fn)[0]
            try:
                return int(base)
            except:
                return base
        frame_list.sort(key=key_fn)
        frames = [os.path.join(seq_dir, f) for f in frame_list]

        if len(frames) == 0:
            raise RuntimeError(f'找不到图像帧: {sequence_name} 于 {seq_dir}')

        # 读取标注，兼容逗号或空白
        if not os.path.isfile(anno_path):
            raise RuntimeError(f'找不到标注文件: {anno_path}')
        gt = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)

        # 长度对齐检查
        if len(gt) != len(frames):
            raise RuntimeError(f'标注与帧数不一致: {sequence_name}, gt={len(gt)}, frames={len(frames)}')

        return Sequence(sequence_name, frames, 'vmat', gt.reshape(-1, 4))