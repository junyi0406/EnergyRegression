import numpy as np 
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras import layers
class Trainer:
    def __init__(self, conf) -> None:
        import pathlib
        import shutil
        self.config = conf
        self.logdir = pathlib.Path(conf["logdir"])/"tensorboard_logs"
        shutil.rmtree(self.logdir, ignore_errors=True)
        
    def get_callbacks(self, name):

        import tensorflow_docs as tfdocs
        return [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
            tf.keras.callbacks.TensorBoard(self.config["logdir"]/name),
        ]
        
    def get_optimizer(self, STEPS_PER_EPOCH):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False)

        return tf.keras.optimizers.Adam(lr_schedule)
  
    # for the beginners
    def build_and_compile_model(self, norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
        return model
        
    # compile different model with optimizer
    def compile_model(self, model, optimizer=None):
        if optimizer is None:
            optimizer = self.get_optimizer()
        model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        metrics=[
                            tf.keras.metrics.BinaryCrossentropy(
                                from_logits=True, name='binary_crossentropy'),
                            'accuracy'])
        model.summary()
        return model 
    
    def fit_model(self, model, name, features, targets, max_epochs=10000 ):
        history = model.fit(
            features,
            targets,
            steps_per_epoch = self.STEPS_PER_EPOCH,
            epochs=max_epochs,
            validation_split=0.5,
            callbacks=self.get_callbacks(name),
            verbose=0)
        return history
    

    def build_model(self, model_type, FEATURES):

        if model_type == "tiny":
            tiny_model = tf.keras.Sequential([
                layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
                layers.Dense(1)
            ])
            return tiny_model
        elif model_type == "small":
            small_model = tf.keras.Sequential([
            # `input_shape` is only required here so that `.summary` works.
            layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(16, activation='elu'),
            layers.Dense(1)])
            return small_model
        elif model_type == "medium":
            medium_model = tf.keras.Sequential([
            layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(64, activation='elu'),
            layers.Dense(64, activation='elu'),
            layers.Dense(1)])
            return medium_model
        elif model_type == "large":
            large_model = tf.keras.Sequential([
            layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(512, activation='elu'),
            layers.Dense(512, activation='elu'),
            layers.Dense(512, activation='elu'),
            layers.Dense(1)])
            return large_model
        else:
            print(f"model_type {model_type} isn't supported!")
            print()
            raise ValueError(f"model_type {model_type} isn't supported!: Only tiny, small, medium, large")



            
        