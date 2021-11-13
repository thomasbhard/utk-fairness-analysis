from utk_utils import *
from utk_generator import UtkFaceDataGenerator
from utk_model import UtkMultiOutputModel
import time
import json
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# r'C:\Users\thoma\Documents\_FAIRALGOS\utk-fairness-analysis\dataset\UTKFace'
# r'C:\Users\thoma\Documents\_FAIRALGOS\utk-fairness-analysis\models'


default_parameters = {
    'IM_WIDTH': 198,
    'IM_HEIGHT': 198,
    'TRAIN_TEST_SPLIT': 0.7,
    'TRAIN_WITH_WEIGTHS': False,
    'INIT_LR': 1e-4,
    'EPOCHS': 1,
    'BATCH_SIZE': 32,
    'BATCH_SIZE_VALID': 32,
    'BATCH_SIZE_TEST': 128,
    'dataset_path': '/content/UTKFaceFull/UTKFace',
    'output_dir': 'content/models',
}



def main(parameters):

    # GLOBAL PARAMETERS
    IM_WIDTH = parameters['IM_WIDTH']
    IM_HEIGHT = parameters['IM_HEIGHT']

    TRAIN_TEST_SPLIT = parameters['TRAIN_TEST_SPLIT']
    TRAIN_WITH_WEIGTHS = parameters['TRAIN_WITH_WEIGHTS']

    INIT_LR = parameters['INIT_LR']
    EPOCHS = parameters['EPOCHS']
    BATCH_SIZE = parameters['BATCH_SIZE']
    BATCH_SIZE_VALID = parameters['BATCH_SIZE_VALID']
    BATCH_SIZE_TEST = parameters['BATCH_SIZE_TEST']

    # -------------------------

    # PATHS AND FILENAMES

    outputdir_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time()))
    outputdir_path = os.path.join(parameters['output_dir'], outputdir_name)
    os.mkdir(outputdir_path)

    # --------------------------


    # SAVE PARAMETERS

    parameters['outputdir_path'] = outputdir_path


    with open(os.path.join(outputdir_path, 'parameters.json'), 'w') as fp:
        json.dump(parameters, fp)

    # ---------------------------

    # PREPROCESSING

    df = parse_dataset(parameters['dataset_path'])

    data_generator = UtkFaceDataGenerator(df, dataset_dict, TRAIN_TEST_SPLIT, IM_WIDTH, IM_HEIGHT, get_weight=None)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

    # ----------------

    # MODEL

    model = UtkMultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT, num_races=len(dataset_dict['race_alias']))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    model.compile(optimizer=opt, 
                loss={
                    'age_output': 'mse', 
                    'race_output': 'categorical_crossentropy', 
                    'gender_output': 'binary_crossentropy'},
                loss_weights={
                    'age_output': 4., 
                    'race_output': 1.5, 
                    'gender_output': 0.1},
                metrics={
                    'age_output': 'mae', 
                    'race_output': 'accuracy',
                    'gender_output': 'accuracy'})


    batch_size = BATCH_SIZE
    valid_batch_size = BATCH_SIZE_VALID
    train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size, include_weights=TRAIN_WITH_WEIGTHS)
    valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size, include_weights=TRAIN_WITH_WEIGTHS)

    callbacks = [
        ModelCheckpoint(os.path.join(outputdir_path, "model_checkpoint"), monitor='val_loss')
    ]

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=len(valid_idx)//valid_batch_size)


    plt.plot(history.history['gender_output_accuracy'])
    plt.plot(history.history['val_gender_output_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig(os.path.join(outputdir_path, 'gender_accuracy.png'))

    # TEST SET

    # make predictions on test set
    test_batch_size = BATCH_SIZE_TEST
    test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size)
    age_pred, race_pred, gender_pred = model.predict_generator(test_generator, 
                                                            steps=len(test_idx)//test_batch_size)


    # collect test set
    test_generator = data_generator.generate_images(test_idx, is_training=False, batch_size=test_batch_size, include_weights=True, include_files=True)
    images, age_true, race_true, gender_true, sample_weights, files = [], [], [], [], [], []
    for test_batch in test_generator:
        image = test_batch[0]
        labels = test_batch[1]
        
        images.extend(image)

        age_true.extend(labels[0])
        race_true.extend(labels[1])
        gender_true.extend(labels[2])

        sample_weights.extend(test_batch[2])
        files.extend(test_batch[3])
        
    # transform labels and predictions
    age_true = np.array(age_true)
    race_true = np.array(race_true)
    gender_true = np.array(gender_true)

    race_true, gender_true = race_true.argmax(axis=-1), gender_true.argmax(axis=-1)
    race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)

    age_true = age_true * data_generator.max_age
    age_pred = age_pred * data_generator.max_age
    age_pred_flat = age_pred.flatten()


    # OUTPUT DATAFRAME

    df_prediction = pd.DataFrame({'age_true': age_true, 'age_pred': age_pred_flat, 'race_true': race_true, 'race_pred': race_pred, 'gender_true': gender_true, 'gender_pred': gender_pred})
    df_prediction = df_prediction.round(0).astype(int)
    df_prediction['weights'] = sample_weights
    df_prediction['files'] = files

    df_prediction.to_csv(os.path.join(outputdir_path, 'predictions.csv'))