import mlflow
import tensorflow as tf
from dataset import create_dataset
from model import baseline_model_cnn, build_transfer_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import os
import seaborn as sns
import mlflow
import subprocess
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

SEED = 42
BATCH_SIZE = 32
tf.random.set_seed(SEED)

USE_AUTOLOG = True
RUN_NAME = 'test'
MODEL_TYPE = 'baseline'  # Options: 'baseline', 'transfer', 'finetune'
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
UNFREEZE_CONV5_ONLY = True  # If True, only unfreeze the last conv5 block of ResNet50 during fine-tuning 
OPTIMIZER = 'adam'

username = os.environ.get("DAGSHUB_USERNAME")
token = os.environ.get("DAGSHUB_TOKEN")
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

if username and token and tracking_uri:
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    print("Connected to DagsHub")
else:
    print("DagsHub env vars not found. Logging locally.")

def get_git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:7]

def get_dvc_version():
    return subprocess.check_output(['dvc', 'version']).decode('ascii').strip().split('\n')[0]

class ManualMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        metric_mapping = {
            "loss": "train_loss",
            "val_loss": "val_loss",
            "accuracy": "train_accuracy",
            "val_accuracy": "val_accuracy"
        }
        
        for log_key, mlflow_name in metric_mapping.items():
            val = logs.get(log_key)
            if val is not None:
                mlflow.log_metric(mlflow_name, float(val), step=epoch + 1)

df = pd.read_csv("Artifacts/data_audit.csv")
classes = df['class'].unique()
df['label'] = df['class'].map({classes[0]: 0, classes[1]: 1})

file_paths = df['file path']
labels = df['label']

train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=SEED)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=SEED)

train_paths.to_csv("Artifacts/train_split.csv", index=False)
val_paths.to_csv("Artifacts/val_split.csv", index=False)
test_paths.to_csv("Artifacts/test_split.csv", index=False)

train_ds = create_dataset(train_paths, train_labels, is_train=True)
val_ds   = create_dataset(val_paths, val_labels, is_train=False)
test_ds  = create_dataset(test_paths, test_labels, is_train=False)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
modelCheckpoints = tf.keras.callbacks.ModelCheckpoint(filepath="Artifacts/best_model_checkpoints.keras", monitor='val_loss', save_best_only=True)
learningRateReduction = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

callbacks_list = [earlyStopping, modelCheckpoints, learningRateReduction]

if not USE_AUTOLOG:
    callbacks_list.append(ManualMetricsCallback())
else:
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False) # type: ignore

mlflow.set_experiment("MLOPS_A2_Training")
with mlflow.start_run(run_name=RUN_NAME):   
    mlflow.set_tag("git_commit", get_git_commit())
    mlflow.set_tag("dvc_version", get_dvc_version())
    mlflow.set_tag("data_version", 1.0) 
    mlflow.set_tag("seed", SEED)
    mlflow.set_tag("logging_method", "autolog" if USE_AUTOLOG else "manual")
        
    mlflow.log_param("model_type", MODEL_TYPE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("image_size", IMAGE_SIZE)
    
    mlflow.log_artifact("Artifacts/train_split.csv")
    mlflow.log_artifact("Artifacts/val_split.csv")
    mlflow.log_artifact("Artifacts/test_split.csv")
    if os.path.exists("requirements.txt"):
        mlflow.log_artifact("requirements.txt")
    
    if MODEL_TYPE == 'baseline':
        model = baseline_model_cnn()
    else:
        transfer_model, base_model = build_transfer_model()
        if MODEL_TYPE == "finetune":
            base_model.trainable = True
            if UNFREEZE_CONV5_ONLY:
                for layer in base_model.layers:
                    layer.trainable = True if "conv5" in layer.name else False
            else:
                pass
            
        transfer_model.compile(
            optimizer=tf.keras.optimizers.deserialize({
                "class_name": OPTIMIZER,
                "config": {"learning_rate": LEARNING_RATE}
            }),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model = transfer_model
        
    print(f"\nTraining {MODEL_TYPE}")
    model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks_list)
    model.save(f'Artifacts/{MODEL_TYPE}_cnn.keras')
    
    if not USE_AUTOLOG:
        mlflow.keras.log_model(model, "trained_model") # type: ignore

    model = tf.keras.models.load_model(f'Artifacts/{MODEL_TYPE}_cnn.keras')

    val_prob = model.predict(val_ds).flatten()
    val_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()

    test_probs = model.predict(test_ds).flatten()
    test_true = np.concatenate([y for x, y in test_ds], axis=0).flatten()

    threshold = 0.5
    best_f1 = 0

    for thresh in np.arange(0.1, 0.9, 0.001):
        val_pred = (val_prob > thresh).astype(int)
        current_f1 = f1_score(val_true, val_pred) 

        if best_f1 < current_f1:
            best_f1 = current_f1
            threshold = thresh
            
    mlflow.log_metric("best_threshold", float(threshold))
    print(f"Best threshold found: {threshold:.2f} (F1-Score: {best_f1:.4f})")
    test_pred = (test_probs > threshold).astype(int)
    
    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred)
    recall = recall_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred)
    roc_auc = roc_auc_score(test_true, test_probs)
    
    mlflow.log_metric("test_accuracy", float(accuracy))
    mlflow.log_metric("test_precision", float(precision))
    mlflow.log_metric("test_recall", float(recall))
    mlflow.log_metric("test_f1_score", float(f1))
    mlflow.log_metric("test_roc_auc", float(roc_auc))

    print("\nTest Set Evaluation:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)
    print("ROC-AUC: ", roc_auc)

    cm = confusion_matrix(test_true, test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("Artifacts/cm.png")
    plt.close()
    mlflow.log_artifact("Artifacts/cm.png")

    fpr, tpr, thresholds = roc_curve(test_true, test_probs)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(test_true, test_probs)}')
    plt.plot([0, 1], [0, 1], 'k--')
    idx = np.argmin(np.abs(thresholds - threshold))
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Threshold ({threshold})')

    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("Artifacts/roc_curve.png")
    plt.close()
    mlflow.log_artifact("Artifacts/roc_curve.png")

    misclassified_images = []

    for path, true_label, pred_label, prob in zip(test_paths, test_true, test_pred, test_probs):
        if true_label != pred_label:
            misclassified_images.append({
                'path': path,
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_prob': prob
            })

    top_10_errors = misclassified_images[:10]
    print(f"\nFound {len(misclassified_images)} total errors. Plotting the first 10...")

    if len(top_10_errors) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten() 

        for i, err in enumerate(top_10_errors):
            img = tf.keras.utils.load_img(err['path'])
            axes[i].imshow(img)

            true_name = "FAKE" if err['true_label'] == 1 else "REAL"
            pred_name = "FAKE" if err['pred_label'] == 1 else "REAL"

            axes[i].set_title(f"True: {true_name}\nPred: {pred_name}\nProb: {err['pred_prob']:.2f}")
            axes[i].axis('off') 

        plt.suptitle("Misclassified Test Images")
        plt.tight_layout()
        fig = plt.gcf() 
        mlflow.log_figure(fig, f"{MODEL_TYPE}_errors.png")
        plt.close(fig) 
