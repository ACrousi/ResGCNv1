import pickle
import logging
import numpy as np
from torch.utils.data import Dataset

from .data_utils import multi_input

logger = logging.getLogger()


class Feeder(Dataset):
    def __init__(self, phase, path=None, data_shape=None, connect_joint=None, debug=False,
                 data_path=None, label_path=None, eval_data_path=None, eval_label_path=None, **kwargs):
        self.split = phase
        self.conn = connect_joint if connect_joint is not None else []
        
        if phase == 'train':
            self.data_path = data_path or f"{path}/train_data.npy"
            self.label_path = label_path or f"{path}/train_label.pkl"
        elif phase == 'eval':
            self.data_path = eval_data_path or f"{path}/val_data.npy"
            self.label_path = eval_label_path or f"{path}/val_label.pkl"
        else:
            raise ValueError(f"Invalid phase: {phase}")

        self.load_data()

    def load_data(self):
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        # 原始形狀: (3, 150, 17, 1)
        # 使用 multi_input 函數生成三流輸入: (3, 6, 150, 17, 1)
        data_numpy = multi_input(data_numpy, self.conn)
        
        label = self.label[index]
        name = self.sample_name[index] if hasattr(self, 'sample_name') else str(index)

        return data_numpy, label, name