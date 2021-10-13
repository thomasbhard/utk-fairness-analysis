import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from PIL import Image


class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df, dataset_dict, train_test_split, im_width, im_height, get_weight):
        self.df = df
        self.dataset_dict = dataset_dict
        self.train_test_split = train_test_split
        self.im_width = im_width
        self.im_height = im_height
        if get_weight is not None:
          self.get_weight = get_weight
        else:
          # if there is no function for get_weight specified return 1 for all samples
          self.get_weight = lambda g,r,a: 1
        
    def generate_split_indexes(self):
        p = np.random.RandomState(seed=42).permutation(len(self.df))
        train_up_to = int(len(self.df) * self.train_test_split)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * self.train_test_split)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: self.dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: self.dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((self.im_width, self.im_height))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16, include_weights=False, include_files=False):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages, races, genders = [], [], [], []

        # weights
        sample_weights = []
        files = []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']
                
                im = self.preprocess_image(file)
                
                ages.append(age / self.max_age)
                races.append(to_categorical(race, len(self.dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(self.dataset_dict['gender_id'])))
                images.append(im)
                
                if include_weights:
                  sample_weights.append(self.get_weight(gender, age, race))

                if include_files:
                  files.append(file)

                # yielding condition
                if len(images) >= batch_size:
                    if include_files and include_weights:
                      yield np.array(images), [np.array(ages), np.array(races), np.array(genders)], np.array(sample_weights), np.array(files)
                    
                    elif include_files:
                      yield np.array(images), [np.array(ages), np.array(races), np.array(genders)], np.array(files)

                    elif include_weights:
                      yield np.array(images), [np.array(ages), np.array(races), np.array(genders)], np.array(sample_weights)

                    else:
                      yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]

                    images, ages, races, genders, sample_weights, files = [], [], [], [], [], []
                    
            if not is_training:
                break


if __name__ == "__main__":
  pass
  # data_generator = UtkFaceDataGenerator(df)
  # train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()