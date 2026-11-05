import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import create_dataset

SEED = 42
BATCH_SIZE = 32

tf.random.set_seed(SEED)

df = pd.read_csv("Artifacts/data_audit.csv")
classes = df['class'].unique()
df['label'] = df['class'].map({classes[0]: 0, classes[1]: 1})

file_paths = df['file path']
labels = df['label']

train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=SEED)
val_paths, _, val_labels, _ = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=SEED)

train_ds = create_dataset(train_paths, train_labels, is_train=True)
val_ds   = create_dataset(val_paths, val_labels, is_train=False)

def model_tune(hp):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = True
    
    num_frozen_layers = hp.Int('num_unfrozen_layers', min_value=0, max_value=30, step=5)
    for layer in base_model.layers[-num_frozen_layers:]:
        layer.trainable = True
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
        
    x = layers.GlobalAveragePooling2D()(x)
    
    dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    
    lr = hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model 

tuner = kt.RandomSearch(
    model_tune,
    objective='val_accuracy',
    max_trials=8,        
    directory='vg_tuner_dir',
    project_name='visionguard_tune',
    overwrite=True        
)

tuner.search(train_ds, epochs=5, validation_data=val_ds)
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Learning rate: ", best_hp.get('learning_rate'))
print("Dropout rate: ", best_hp.get('dropout'))
print("Number of unfrozen layers: ", best_hp.get('num_unfrozen_layers'))