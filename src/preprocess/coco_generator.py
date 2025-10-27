import os
import pickle
import logging
import numpy as np
from tqdm import tqdm

from .. import utils as U
from .preprocessor import pre_normalization

class COCO_Generator():
    def __init__(self, args, dataset_args):
        self.num_person_out = 2  # Can be adjusted
        self.num_joint = 17      # COCO has 17 joints
        self.max_frame = 300     # Can be adjusted
        self.dataset = 'coco'
        self.print_bar = not args.no_progress_bar
        self.generate_label = args.generate_label

        # Define the output path for the processed data
        self.out_path = '{}/coco'.format(dataset_args['path'])
        U.create_folder(self.out_path)

        # Define paths to your raw COCO data
        # IMPORTANT: You need to modify these paths to match your data structure
        self.coco_data_path = dataset_args.get('coco_data_path', './data/coco/raw_data')
        self.coco_label_path = dataset_args.get('coco_label_path', './data/coco/raw_data_labels.json')

        # Get skeleton file list
        # This part needs to be adapted to how your COCO data is stored
        self.file_list = sorted(os.listdir(self.coco_data_path))

    def start(self):
        # Generate data for train and eval phases
        for phase in ['train', 'eval']:
            logging.info(f'Phase: {phase}')
            self.gendata(phase)

    def read_xyz(self, file_path, score_threshold=0.1):
        # This function assumes you have a way to load your data into a numpy array.
        # The loaded data should have a shape like (M, T, V, C)
        # M: number of persons
        # T: number of frames
        # V: number of joints (17)
        # C: number of channels (3 for x, y, score)
        
        #
        # --- IMPORTANT ---
        # You need to replace the following line with your actual data loading logic.
        # For example: `raw_data = np.load(file_path)` or custom parsing.
        # Let's assume `raw_data` is a numpy array with shape (num_persons, num_frames, 17, 3)
        #
        # Placeholder for raw data:
        raw_data = np.random.rand(self.num_person_out, self.max_frame, self.num_joint, 3) # M, T, V, C
        
        # Separate coordinates and scores
        xy_data = raw_data[..., :2]  # Shape: (M, T, V, 2)
        scores = raw_data[..., 2]    # Shape: (M, T, V)
        
        # Apply score threshold: if score is low, set coordinates to 0
        mask = scores < score_threshold
        xy_data[mask] = 0
        
        # Normalize coordinates to [-1, 1] range
        # Assuming original image size is 1080x1920 (width x height)
        image_width = 1080
        image_height = 1920
        
        # Create a mask for valid (non-zero) coordinates
        valid_mask = ~mask
        
        # Normalize x coordinates (width dimension)
        xy_data[..., 0] = np.where(valid_mask,
                                  (xy_data[..., 0] / image_width) * 2 - 1,
                                  xy_data[..., 0])
        
        # Normalize y coordinates (height dimension)
        xy_data[..., 1] = np.where(valid_mask,
                                  (xy_data[..., 1] / image_height) * 2 - 1,
                                  xy_data[..., 1])
        
        # Create the final data tensor with shape (C, T, V, M)
        # C=3 for (x, y, z), where z is always 0.
        num_persons, num_frames, num_joints, _ = raw_data.shape
        data = np.zeros((3, num_frames, num_joints, num_persons), dtype=np.float32)
        
        # Fill in x and y coordinates
        data[:2, :, :, :] = xy_data.transpose(3, 1, 2, 0) # Transpose from (M,T,V,2) to (2,T,V,M)
        
        return data

    def gendata(self, phase):
        # IMPORTANT: This logic needs to be adapted to your dataset split.
        # You need a way to distinguish between training and evaluation samples.
        # This could be based on file names, a separate list, or folder structure.

        # Placeholder logic: 80% for training, 20% for evaluation
        num_samples = len(self.file_list)
        train_split = int(num_samples * 0.8)
        
        if phase == 'train':
            sample_files = self.file_list[:train_split]
        elif phase == 'eval':
            sample_files = self.file_list[train_split:]
        else:
            raise ValueError(f'Unknown phase: {phase}')

        sample_names = []
        sample_labels = []
        
        # This is a placeholder. You need to load your actual labels.
        # For example, from a JSON or text file.
        # We'll just assign a dummy label for now.
        for i, file_name in enumerate(sample_files):
            sample_names.append(file_name)
            sample_labels.append(i % 10) # Dummy labels (e.g., 10 classes)

        # Save labels
        with open(f'{self.out_path}/{phase}_label.pkl', 'wb') as f:
            pickle.dump((sample_names, list(sample_labels)), f)

        if not self.generate_label:
            # Create data tensor (N, C, T, V, M)
            fp = np.zeros((len(sample_labels), 3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)

            # Fill data tensor
            items = tqdm(sample_files, dynamic_ncols=True) if self.print_bar else sample_files
            for i, file_name in enumerate(items):
                file_path = os.path.join(self.coco_data_path, file_name)
                data = self.read_xyz(file_path)
                fp[i, :, 0:data.shape[1], :, :] = data

            # Perform preprocessing on data tensor
            # fp = pre_normalization(fp, print_bar=self.print_bar)

            # Save input data
            np.save(f'{self.out_path}/{phase}_data.npy', fp)
