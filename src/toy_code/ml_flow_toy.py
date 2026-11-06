import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

mlflow.set_experiment("fashion_mnist_toy")
mlflow.tensorflow.autolog() # type: ignore

with mlflow.start_run(run_name = "run_1_fashion_mnist"):
    model1 = build_model(learning_rate=0.01)
    history1 = model1.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
    
    plt.plot(history1.history['loss'], label='Train Loss')
    plt.plot(history1.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve (LR=0.01)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot_run1.png')
    plt.close()
    
    mlflow.log_artifact('loss_plot_run1.png')
    
with mlflow.start_run(run_name ="run_2_fashion_mnist"):
    model2 = build_model(learning_rate=0.001)
    history2 = model2.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)
    
    plt.plot(history2.history['loss'], label='Train Loss')
    plt.plot(history2.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve (LR=0.001)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot_run2.png')
    plt.close()
    
    mlflow.log_artifact('loss_plot_run2.png')