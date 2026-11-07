import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import Hyperband
import pandas as pd
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        
        keras.layers.Dense(
            units=hp.Int('units', min_value=64, max_value=512, step=64),
            activation='relu'
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if os.path.exists('toy_tuning_dir'):
    import shutil
    shutil.rmtree('toy_tuning_dir')

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=15,      
    factor=3,           
    directory='toy_tuning_dir',
    project_name='fashion_mnist_toy',
    overwrite=True
)

tuner.search(
    x_train, y_train, 
    epochs=15, 
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_lr = best_hps.get('learning_rate')
best_units = best_hps.get('units')

print("BEST HYPERPARAMETERS FOUND:")
print(f"Optimal Learning Rate: {best_lr}")
print(f"Optimal Hidden Units:  {best_units}")

trials_data = []
for trial_id, trial in tuner.oracle.trials.items():
    best_val_acc = trial.metrics.get_best_value('val_accuracy')
    
    trials_data.append({
        'trial_id': trial_id,
        'learning_rate': trial.hyperparameters.get('learning_rate'),
        'hidden_units': trial.hyperparameters.get('units'),
        'best_val_accuracy': round(best_val_acc, 4) if best_val_acc else 0.0,
        'status': trial.status
    })

results_df = pd.DataFrame(trials_data)
results_df = results_df.sort_values(by='best_val_accuracy', ascending=False)

csv_filename = 'Artifacts/toy_results_table.csv'
results_df.to_csv(csv_filename, index=False)

print(f"Toy results table successfully saved to '{csv_filename}'")
print("\nTop 5 Trials Preview:")
print(results_df.head())