import pickle
import numpy as np

# 載入標籤檔案
with open('./data/coco/train_label.pkl', 'rb') as f:
    labels = pickle.load(f)

print('標籤檔案結構:', type(labels))
print('樣本數:', len(labels))
print('前幾個標籤:', labels[:5])

# 嘗試不同的方式來獲取唯一類別
try:
    if isinstance(labels, list):
        if len(labels) > 0:
            first_item = labels[0]
            print('第一個標籤的類型:', type(first_item))
            
            if isinstance(first_item, (int, np.integer)):
                # 標籤是整數列表
                unique_labels = set(labels)
                print('唯一類別數:', len(unique_labels))
                print('類別範圍:', min(labels), 'to', max(labels))
            elif isinstance(first_item, (list, np.ndarray)):
                # 標籤是二維結構
                flat_labels = []
                for item in labels:
                    if isinstance(item, (list, np.ndarray)):
                        flat_labels.extend(item)
                    else:
                        flat_labels.append(item)
                unique_labels = set(flat_labels)
                print('唯一類別數:', len(unique_labels))
                print('類別範圍:', min(flat_labels), 'to', max(flat_labels))
    elif isinstance(labels, np.ndarray):
        if labels.ndim == 1:
            unique_labels = set(labels)
            print('唯一類別數:', len(unique_labels))
            print('類別範圍:', labels.min(), 'to', labels.max())
        else:
            flat_labels = labels.flatten()
            unique_labels = set(flat_labels)
            print('唯一類別數:', len(unique_labels))
            print('類別範圍:', flat_labels.min(), 'to', flat_labels.max())
            
except Exception as e:
    print('處理標籤時發生錯誤:', e)

# 同時檢查資料檔案的形狀
data = np.load('./data/coco/train_data.npy')
print('\n資料檔案形狀:', data.shape)