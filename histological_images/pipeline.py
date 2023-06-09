# System
import os
import datetime as dt
import multiprocessing
# Data manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress, pearsonr
# Deep learning
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import Conv2D
from keras.applications import Xception, VGG16, VGG19
from keras.optimizers import Adam
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Personal tools
import sys
sys.path.append('src/predicting_age_from_tissue_images/')
from utils.normalize_HnE import norm_HnE

MAIN_COLOR = '#808080'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if tf.config.list_physical_devices('GPU'):
    print(">> Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    strategy = tf.distribute.MirroredStrategy()
    print('>> Number of devices withing strategy: {}'.format(strategy.num_replicas_in_sync))
else:
    print(">> No GPUs found")
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
    
def get_tile_images_path(
    list_of_sids, 
    list_of_ages,
    smoker_cats,
    imgs_path_,
    tissue_type,
    experiment_folder,
    dropna=True):
    
    """
    This function takes in various parameters and returns a pandas dataframe containing information
    about tile images.
    
    Args:
      list_of_sids: A list of sample IDs (strings)
      list_of_ages: `list_of_ages` is a list of ages corresponding to each sample ID in `list_of_sids`.
    It is used to create a column in the output dataframe indicating the age of each sample.
      smoker_cats: `smoker_cats` is a list of smoker categories for each sample in `list_of_sids`. It is
    used to assign a smoker status to each tile path in the output dataframe.
      imgs_path_: The path to the folder containing the images of the tissue tiles. It is a string that
    contains two placeholders: tissue_type and experiment_folder. These placeholders will be replaced by
    the actual tissue type and experiment folder names when the function is called.
      tissue_type: The type of tissue being analyzed.
      experiment_folder: `experiment_folder` is a list of strings representing the names of the
    experiment folders where the tile images are stored. The function will search for tile images in
    each of these folders.
      dropna: `dropna` is a boolean parameter that determines whether to drop rows with missing values
    (NaN) from the resulting dataframe. If `dropna=True`, rows with missing values will be dropped,
    otherwise they will be kept in the dataframe. Defaults to True
    
    Returns:
      a pandas DataFrame containing information about the tile images for a given list of sample IDs,
    ages, and smoker categories. The DataFrame includes columns for the sample ID, tile image path, age,
    and smoker status. The function also takes in arguments for the path to the image files, the tissue
    type, and the experiment folder. If multiple experiment folders are provided, the function
    concatenates
    """
    
    dfs = []
    print(experiment_folder)
    for exp_f in experiment_folder:
        imgs_path = imgs_path_.format(tissue_type, exp_f)
        lt_tiles_samples = []
        
        for sid, age, sm_status in zip(list_of_sids, list_of_ages, smoker_cats):
            try:
                image_files = os.listdir(f"{imgs_path}/{sid}/{sid}_tiles")
                if image_files:
                    tile_paths = [f"{imgs_path}/{sid}/{sid}_tiles/{i}" for i in image_files]
                    lt_tiles_samples.append(
                        list(
                            zip(
                                [sid]*len(tile_paths),
                                tile_paths,
                                [age]*len(tile_paths),
                                [sm_status]*len(tile_paths)
                                )
                            )
                        )
                else:
                    # sid with 0 tiles (it can happen when content_theeshold is not reached)
                    lt_tiles_samples.append([(sid, np.nan, age, sm_status)])
            except:
                # tiles with download problems (it can happen when GTEx don't have images an sid)
                lt_tiles_samples.append([(sid, np.nan, age, sm_status)])
        
        tp_tile_paths = tuple(sum(lt_tiles_samples, []))
        df_out = pd.DataFrame(tp_tile_paths, columns=['sample_id', 'tile_path', 'age', 'smoker_status'])
        
        if dropna:
            df_out = df_out.dropna(subset=['tile_path'])
            
        if df_out.empty:
            raise Exception(
                f'''
                Empty dataframe. The function could not find any tile inside {imgs_path}.
                Check this folder. If there are files there, then check the list_of_sids
                you are using to ensure that they are the same as those found in this folder.
                '''
                )
            
        dfs.append(df_out)
        
    assert len(dfs) == len(experiment_folder)
    
    if len(experiment_folder) > 1:
        experiment_folder_name = '_'.join(experiment_folder)
        for d in dfs:
            assert not d.empty
        df_out_new = pd.concat(dfs, axis=0).copy()
    else:
        experiment_folder_name = experiment_folder[0]
        df_out_new = df_out.copy()
    return df_out_new

def get_sample_weights(df_samples, alpha):
    """
    This function calculates sample weights based on the inverse of age frequencies and returns a
    dataframe with the weights column added.
    
    Args:
      df_samples: a pandas DataFrame containing the samples with their corresponding ages
      alpha: Alpha is a hyperparameter that controls the degree of penalization for imbalanced classes.
    A higher value of alpha will result in higher penalization for the more frequent classes, while a
    lower value of alpha will result in less penalization.
    
    Returns:
      a pandas DataFrame with an additional column 'weights' that contains the sample weights calculated
    based on the inverse of age frequencies. The 'age_groups' column is dropped before returning the
    DataFrame.
    """
    df_samples['age_groups'] = pd.cut(df_samples['age'], 6)
    y_train = df_samples['age_groups'].values
    
    import numpy as np
    unique_ages, age_freq = np.unique(y_train, return_counts=True)
    age_weights = {age: 1 / (freq ** alpha) for age, freq in zip(unique_ages, age_freq)}

    sample_weights = np.array([age_weights[age] for age in y_train])
    df_samples['weights'] = sample_weights
    return df_samples.drop('age_groups', axis=1)

def setup_model(model_type, input_shape):
    """
    This function sets up and returns various types of pre-defined deep learning models for image
    classification tasks.
    
    Args:
      model_type: The type of model to be set up. It can be one of the following: 'cnn_01', 'cnn_02',
    'xception', 'vgg16', 'vgg19', 'vgg19_mod'.
      input_shape: The shape of the input data for the model, which is a tuple of integers representing
    the dimensions of the input data. For example, (224, 224, 3) represents an input image with height
    and width of 224 pixels and 3 color channels (RGB).
    
    Returns:
      The function `setup_model` returns a Keras model based on the specified `model_type` and
    `input_shape`. The returned model is either a custom CNN (`cnn_01` or `cnn_02`), Xception, VGG16,
    VGG19, or a modified VGG19 model.
    """

    if model_type == 'cnn_01':
        model = keras.Sequential()
        # First block
        model.add(layers.Conv2D(filters=32, kernel_size=[3, 3], input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(Conv2D(filters=32, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=32, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]))

        # Second block
        model.add(layers.Conv2D(filters=64, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=64, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=64, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]))

        # Third block
        model.add(layers.Conv2D(filters=128, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=128, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(filters=128, kernel_size=[3, 3]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]))

        # Final block
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(filters=128, kernel_size=[1, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(filters=32, kernel_size=[1, 1]))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        # Flatten and output layer
        model.add(layers.Flatten())
        model.add(layers.Dense(units=1, activation='linear'))
        return model
    
    elif model_type == 'cnn_02':
        model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="linear"),
        ])
        return model

    elif model_type == 'xception':
        base_model = Xception(input_shape=input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(1)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    elif model_type == 'vgg16':
        base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(1)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    elif model_type == 'vgg19':
        base_model = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(1)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    elif model_type == 'vgg19_mod':
        base_model = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
    
        # Fine-tune the last few convolutional layers
        for layer in base_model.layers[:-5]:
            layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        predictions = tf.keras.layers.Dense(1)(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    else:
        raise ValueError('Invalid model type')
 
def setup_batch_generator(data, image_size, batch_size, subset, augmentation=None, weights=None):
    """
    This function sets up and returns a batch generator for training, validation, or testing data with
    optional data augmentation and weighting.
    
    Args:
      data: The input data in the form of a Pandas DataFrame containing the file paths to the image
    tiles and their corresponding age labels.
      image_size: The size of the input images to the model.
      batch_size: The number of samples in each batch of data that will be fed to the model during
    training.
      subset: The subset parameter specifies which subset of the data to generate batches for. It can be
    either 'train', 'valid', or 'test'.
      augmentation: A boolean parameter that determines whether data augmentation should be applied to
    the training data. If True, the function applies various transformations to the images in the
    training set to increase the size and diversity of the training data. If False, the function only
    rescales the pixel values of the images.
      weights: The weights parameter is used to assign a weight to each sample in the dataset. This can
    be useful when dealing with imbalanced datasets, where some classes have significantly fewer samples
    than others. By assigning a higher weight to the underrepresented samples, the model can be trained
    to give them more importance during training
    
    Returns:
      a generator object for the specified subset of the data (train, validation, or test). The
    generator object is created using the specified data, image size, batch size, subset, augmentation,
    and weights (if provided). The generator object is used to generate batches of image data and
    corresponding labels during model training or evaluation.
    """
    if subset == 'train':
        if augmentation:
            datagen_train = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.08,
                height_shift_range=0.08,
                # shear_range=0.1,
                # zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='wrap'
            )
        else:
            datagen_train = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
            )

        # Create the train generator
        train_generator = datagen_train.flow_from_dataframe(
            data,
            directory=None,
            x_col="tile_path",
            y_col="age",
            weight_col=weights,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="raw"
        )
        
        return train_generator
    
    elif subset == 'valid':
        datagen_valid = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        # Create the validation generator
        validation_generator = datagen_valid.flow_from_dataframe(
            data,
            directory=None,
            x_col="tile_path",
            y_col="age",
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="raw"
        )
        
        return validation_generator
    
    elif subset == 'test':
        test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_dataframe(
            data,
            directory=None,
            x_col="tile_path",
            y_col="age",
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode="raw",
            shuffle=False  # Important to keep data in the same order as labels
        )

        return test_generator

def apply_function_generator(batch_generator, he_components, weights=False):
    """
    This function takes in a batch generator and applies a normalization function to the images in the
    batch, returning either the H (Hematoxylin), E (Eosin), or normalized components along with the batch labels and weights
    (if specified).
    
    Args:
      batch_generator: A generator that yields batches of input data and corresponding labels.
      he_components: he_components is a string parameter that specifies which component of the H&E
    stained image to use for training. It can take one of three values: 'H' for the Hematoxylin
    component, 'E' for the Eosin component, or 'norm' for the normalized image.
      weights: `weights` is a boolean parameter that determines whether or not to include the weights in
    the output. If `weights` is True and the batch generator includes weights, then the output will
    include the weights as well. If `weights` is False or the batch generator does not include weights,
    then the. Defaults to False
    """
    for batch in batch_generator:
        lt_norm_img = []
        H_img = []
        E_img = []

        batch_x, batch_y = batch[0], batch[1]
        batch_weights = batch[2] if len(batch) > 2 else None
        
        for image in batch_x:
            norm_img, h_img, e_img = norm_HnE(image, Io=240, alpha=1, beta=0.15)
            lt_norm_img.append(norm_img)
            H_img.append(h_img)
            E_img.append(e_img)

        if he_components == 'H':
            output = (np.array(H_img), batch_y)
        elif he_components == 'E':
            output = (np.array(E_img), batch_y)
        elif he_components == 'norm':
            output = (np.array(lt_norm_img), batch_y)
        
        if weights and batch_weights is not None:
            yield output + (batch_weights,)
        else:
            yield output
            
def get_subsample(df_tiles_path, subsample_size):
    """
    This function takes a dataframe of tile paths and a subsample size, and returns a subsampled version
    of the dataframe grouped by sample ID.
    
    Args:
      df_tiles_path: The input dataframe containing information about image tiles, including the sample
    ID and file path.
      subsample_size: subsample_size is the number of samples to randomly select from each group
    (grouped by 'sample_id') in the input dataframe df_tiles_path. The function returns a new dataframe
    with the same columns as df_tiles_path, but with a reduced number of rows based on the
    subsample_size parameter.
    
    Returns:
      The function `get_subsample` returns a subsampled version of the input dataframe `df_tiles_path`.
    The subsampling is done by randomly selecting a specified number of rows (`subsample_size`) from
    each group of rows that share the same value in the `sample_id` column. The output dataframe has the
    same columns as the input dataframe, and the rows are shuffled within each group.
    """
    lt_columns = list(df_tiles_path.columns)
    df_tiles_path_resamp = (
        df_tiles_path
            .groupby(['sample_id'])
            .apply(lambda x: x.sample(frac=1).sample(min(len(x), subsample_size)))
            .reset_index(drop=True)
            )
    return df_tiles_path_resamp[lt_columns]

def compute_metrics(y_true, y_pred):
    """
    The function computes various metrics such as R2 score, RMSE, slope, intercept, and correlation
    between true and predicted values.
    
    Args:
      y_true: The true values of the target variable (or dependent variable) in a regression problem.
    These are the actual values that we are trying to predict.
      y_pred: The predicted values of the target variable.
    
    Returns:
      The function `compute_metrics` returns a dictionary containing the following metrics: R-squared
    (r2), root mean squared error (rmse), slope, intercept, and correlation (cor). These metrics are
    computed based on the input true values (y_true) and predicted values (y_pred) using various
    statistical functions such as r2_score, mean_squared_error, linregress, and pearson
    """
    metrics = {}
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    metrics['slope'] = slope
    metrics['intercept'] = intercept
    correlation, _ = pearsonr(y_true, y_pred)
    metrics['cor'] = correlation
    return metrics

def plot_model_fit(y_true, y_pred, figure_path, type="Test", smoker_status=None, title_override=None):
    """
    This function plots a scatterplot of predicted vs true values and computes metrics such as r², rmse,
    slope, intercept, and correlation.
    
    Args:
      y_true: The true values of the target variable.
      y_pred: The predicted values of the model.
      figure_path: The file path where the plot will be saved as a PNG file.
      type: The type of data being plotted, either "Test" or "Train". Defaults to Test
      smoker_status: `smoker_status` is a variable that contains information about the smoking status of
    the individuals in the dataset. It is used to color-code the scatterplot points based on whether the
    individual is a smoker or not. If `smoker_status` is not provided, the scatterplot points will not
    be
      title_override: A string that overrides the default title of the plot. If None, the default title
    will be used.
    """
    metrics = compute_metrics(y_true, y_pred)

    jointgrid = sns.jointplot(x=y_true, y=y_pred,
                              kind="reg",
                              truncate=False,
                              scatter=False, fit_reg=True,
                              xlim=(20, 70),
                              ylim=(20, 70),
                              color=MAIN_COLOR
                              )
    
    jointgrid.ax_joint.axline([0, 0], [1, 1], transform=jointgrid.ax_joint.transAxes,
                              linestyle="--", alpha=0.8, color="darkgray")
    sns.scatterplot(x=y_true, y=y_pred, hue=smoker_status, ax=jointgrid.ax_joint, color=MAIN_COLOR)
    if smoker_status is not None:
        sns.move_legend(jointgrid.ax_joint, "lower right")

    if title_override:
        plt.title(title_override)
    else:
        plt.title(f"Xception Model Fit Scatterplot {type} (N=45)")
    jointgrid.ax_joint.set_ylabel("Predicted Values")
    jointgrid.ax_joint.set_xlabel("True Value")
    t = plt.text(.05, .8,
                 'r²={:.3f}\nrmse={:.3f}\nslope={:.3f}\nintercept={:.3f}\ncor={:.3f}'.format(
                     metrics["r2"], metrics["rmse"], metrics["slope"], metrics["intercept"], metrics["cor"]),
                 transform=jointgrid.ax_joint.transAxes)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='darkblue'))
    jointgrid.fig.subplots_adjust(top=0.95)
    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_path, format='png', dpi=100, bbox_inches = "tight")
    plt.close('all')

def stratify_train_test_split(df, stratify_col, test_size=0.25):
    """
    The function performs a stratified train-test split on a pandas dataframe, accounting for categories
    with single instances. If any category has only 1 instance then it will remain in the training set.
    
    Args:
      df: The input dataframe that needs to be split into train and test sets.
      stratify_col: The column in the dataframe that we want to use for stratification during the
    train-test split.
      test_size: test_size is a float value that represents the proportion of the dataset that should be
    allocated to the test set. For example, if test_size=0.25, then 25% of the data will be used for
    testing and 75% will be used for training.
    
    Returns:
      a tuple containing two dataframes: the training data and the testing data. The training data is a
    combination of the multi-instance data that has been stratified using the specified column and the
    single-instance data. The testing data is the remaining multi-instance data that has also been
    stratified using the specified column.
    """
    # Check if stratify column contains single instance categories
    stratify_col_counts = df[stratify_col].value_counts()
    single_instance_ages = stratify_col_counts[stratify_col_counts == 1].index.tolist()
    
    # Separate data with single instance ages
    single_instance_data = df[df[stratify_col].isin(single_instance_ages)]
    multi_instance_data = df[~df[stratify_col].isin(single_instance_ages)]

    # Perform stratified train test split on data with multiple instance ages
    train_multi, test_multi = train_test_split(multi_instance_data, 
                                               test_size=test_size, 
                                               stratify=multi_instance_data[stratify_col],
                                               random_state=42)
    # Add single instance data to training set
    train_data = pd.concat([train_multi, single_instance_data])
    
    return train_data, test_multi

def main(**kwargs):
    """
    Main function to run the deep learning pipeline for age prediction from histological images.

    Parameters:
    **kwargs: Keyword arguments. Valid options include:
        validation_set (bool): Whether to use a validation set during training. Default is True.
        tissue_type (str): Type of tissue for images. Default is 'Lung'.
        imgs_path (str): Path to the images. Default is 'data/01_raw/images/{}/{}/output'.
        experiment_folder (list): List containing name of the experiment folder. Default is ['ct95-ds2-ts256'].
        model_type (str): Type of deep learning model to use. Default is 'xception'.
        tile_size (int): Size of the tiles for the images. Default is 256.
        batch_size (int): Batch size for training. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        epochs (int): Number of epochs for training. Default is 6.
        subsample_valid_size (int): Number of samples to use from the validation set. Default is 100.
        subsample_train_size (int): Number of samples to use from the training set. Default is 100.
        batch_size_test (int): Batch size for testing. Default is 2000.
        augmentation (bool): Whether to use image augmentation during training. Default is False.
        weighted (bool): Whether to use sample weights during training. Default is False.
        he_components (bool): Whether to use HE components for image processing. Default is False.

    Pipeline:
    1. Load the training and testing data.
    2. If a validation set is used, split the training data into a new training set and a validation set.
    3. For each dataset (training, validation, testing), obtain the path of the tile images, and set up the data generator.
    4. Define the model architecture and compile the model with an Adam optimizer and MSE loss.
    5. Fit the model with the training set, using the validation set if it exists.
    6. Make predictions on the test set.
    7. Save the trained model and a scatter plot of the predicted and actual age in the test set.
    """
    
    validation_set = kwargs.get('validation_set', True)
    tissue_type = kwargs.get('tissue_type', 'Lung')
    imgs_path = kwargs.get('imgs_path', 'data/01_raw/images/{}/{}/output')
    experiment_folder = kwargs.get('experiment_folder', ['ct95-ds2-ts256'])
    model_type = kwargs.get('model_type', 'xception')
    tile_size = kwargs.get('tile_size', 256)
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    epochs = kwargs.get('epochs', 6)
    subsample_valid_size = kwargs.get('subsample_valid_size', 100)
    subsample_train_size = kwargs.get('subsample_train_size', 100)
    batch_size_test = kwargs.get('batch_size_test', 2000)
    augmentation = kwargs.get('augmentation', False)
    weighted = kwargs.get('weighted', False)
    he_components = kwargs.get('he_components', False)

    #--------------------------------------
    # Get data
    #--------------------------------------
    df_train = pd.read_csv('data/02_intermediate/Lung/sample_ids_train.csv')
    df_test = pd.read_csv('data/02_intermediate/Lung/sample_ids_test.csv')
    
    # df_train['smoker_status'] = df_train['smoker_status'].fillna('unknown')
    # df_test['smoker_status'] = df_test['smoker_status'].fillna('unknown')
    
    dct_dfs = {}
    dct_dfs['test'] = df_test
    dct_dfs['train'] = df_train
    
    #--------------------------------------
    # Split into train and validation
    #--------------------------------------
    if validation_set:
        df_train, df_valid = stratify_train_test_split(df_train, 'age', 0.35)
        dct_dfs['train'] = df_train
        dct_dfs['valid'] = df_valid
    
    
    #--------------------------------------
    # Get dataframe with tile paths and create batch generator
    #--------------------------------------
    dct_dfs_gens = {}
    for i, j in dct_dfs.items():
        print(f'loading {i}...' )
        df_tiles = get_tile_images_path(
            list_of_sids=j['sample_id'], 
            list_of_ages=j['age'],
            smoker_cats=j['smoker_status'],
            imgs_path_=imgs_path,
            tissue_type=tissue_type,
            experiment_folder=experiment_folder,
            dropna=True
            )
        
        if i == 'train':
            df_tiles = get_subsample(df_tiles, subsample_train_size)
        elif i == 'valid':
            df_tiles = get_subsample(df_tiles, subsample_valid_size)
            
        dct_dfs_gens[f'{i}_tiles'] = df_tiles
        
        bc = batch_size
        if i == 'test':
            bc = batch_size_test
        
        if weighted:
            df_tiles = get_sample_weights(df_tiles, 1)
            weights = 'weights'
        else:
            weights = None
            
        dct_dfs_gens[i] = setup_batch_generator(
            data=df_tiles, 
            image_size=tile_size, 
            batch_size=bc, 
            subset=i, 
            augmentation=augmentation, 
            weights=weights
            )
        
        if he_components:
            if i == 'train':
                dct_dfs_gens[f'{i}_he'] = apply_function_generator(dct_dfs_gens[i], he_components, weights=True)
            else:
                dct_dfs_gens[f'{i}_he'] = apply_function_generator(dct_dfs_gens[i], he_components)
    
    #--------------------------------------
    # Define model
    #--------------------------------------
    with strategy.scope():
        model = setup_model(model_type, input_shape=(tile_size, tile_size, 3))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='mean_squared_error', 
            metrics=[keras.metrics.RootMeanSquaredError()]
            )
                
    # Setup TensorBoard callback
    # log_dir = os.path.join("logs", "fit", time.strftime("%Y%m%d-%H%M%S"))
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    #--------------------------------------
    # Training process
    #--------------------------------------
    if validation_set:
        validation_steps = len(dct_dfs_gens['valid'])
        if he_components:
            valid_gen = dct_dfs_gens['valid_he']
        else:
            valid_gen = dct_dfs_gens['valid']
    else:
        validation_steps = None
        valid_gen = None
        
    if he_components:
        train_gen = dct_dfs_gens['train_he']
    else:
        train_gen = dct_dfs_gens['train']
    
    model.fit(
        train_gen, 
        epochs=epochs,
        steps_per_epoch  = len(dct_dfs_gens['train']),
        validation_steps = validation_steps,
        validation_data  = valid_gen,
        # callbacks=[tensorboard_callback]
        )
    
    #--------------------------------------
    # Predict on test set
    #--------------------------------------
    if he_components:
        train_gen = dct_dfs_gens['test_he']
    else:
        train_gen = dct_dfs_gens['test']
        
    predictions_test = []
    labels_test = []
    for batch_X_test, batch_y_test in train_gen:
        batch_predictions_test = model.predict(batch_X_test)
        predictions_test.extend(batch_predictions_test)
        labels_test.append(batch_y_test)
        if len(predictions_test) >= len(dct_dfs_gens['test_tiles']):
            break
    
    ar_predictions_test = np.concatenate(predictions_test)
    dct_dfs_gens['test_tiles']['pred'] = ar_predictions_test
    df_pred = dct_dfs_gens['test_tiles'].groupby('sample_id').agg({'age':'median', 'pred':'median'})

    #--------------------------------------
    # Save figure and model
    #--------------------------------------
    
    # Create the directory if it does not exist
    timestamp_str = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp_str)
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = os.path.join(log_dir, "model.h5")
    model.save(model_path)
    
    figure_path = os.path.join(log_dir, "test_scatter.png")
    plot_model_fit(df_pred['age'], df_pred['pred'], figure_path)
    
if __name__ == '__main__':
    kwargs = {
        'validation_set': True,
        'tissue_type': 'Lung',
        'imgs_path': 'data/01_raw/images/{}/{}/output',
        'experiment_folder': ['ct95-ds2-ts256'],
        'model_type': 'xception',
        'tile_size': 256,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'epochs': 6,
        'subsample_train_size': 100,
        'subsample_valid_size': 100,
        'batch_size_test': 2000,
        'augmentation': False,
        'weighted': True,
        'he_components': False, #False, H, E, norm
    }

    main(**kwargs)