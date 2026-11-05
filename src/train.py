import tensorflow as tf
from dataset import create_dataset
from model import baseline_model_cnn, build_transfer_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
BATCH_SIZE = 32

tf.random.set_seed(SEED)

df = pd.read_csv("Artifacts/data_audit.csv")
classes = df['class'].unique()
df['label'] = df['class'].map({classes[0]: 0, classes[1]: 1})

file_paths = df['file path']
labels = df['label']

train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=SEED)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=SEED)

train_ds = create_dataset(train_paths, train_labels, is_train=True)
val_ds   = create_dataset(val_paths, val_labels, is_train=False)
test_ds  = create_dataset(test_paths, test_labels, is_train=False)

# earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# modelCheckpoints = tf.keras.callbacks.ModelCheckpoint(filepath="Artifacts/best_model_checkpoints.keras", monitor='val_loss', save_best_only=True)
# learningRateReduction = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

# callbacks_list = [earlyStopping, modelCheckpoints, learningRateReduction]

# baseline_model = baseline_model_cnn()
# history_baseline = baseline_model.fit(
#     train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks_list 
# )

# baseline_model.save('Artifacts/baseline_cnn.keras')
# print("Baseline CNN training complete and saved.")

# transfer_model, base_model = build_transfer_model()
# history_transfer = transfer_model.fit(
#     train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks_list 
# )

# transfer_model.save('Artifacts/transfer_cnn.keras')
# print("Transfer CNN training complete and saved.")

# base_model.trainable = True
# for layer in base_model.layers:
#     if "conv5" in layer.name:
#         layer.trainable = True
#     else:
#         layer.trainable = False
        
# transfer_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss='binary_crossentropy',
#     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
# )

# history_finetune = transfer_model.fit(
#     train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks_list 
# )

# transfer_model.save('Artifacts/finetune_cnn.keras')
# print("FineTune CNN training complete and saved.")

model = tf.keras.models.load_model('Artifacts/finetune_cnn.keras')

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
        
print(f"Best threshold found: {threshold:.2f} (F1-Score: {best_f1:.4f})")

test_pred = (test_probs > threshold).astype(int)

print("\nTest Set Evaluation:")
print("Accuracy: ", accuracy_score(test_true, test_pred))
print("Precision: ", precision_score(test_true, test_pred))
print("Recall: ", recall_score(test_true, test_pred))
print("F1-Score: ", f1_score(test_true, test_pred))
print("ROC-AUC: ", roc_auc_score(test_true, test_probs))

cm = confusion_matrix(test_true, test_pred)
plt.figure(figsize=(5, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

fpr, tpr, thresholds = roc_curve(test_true, test_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(test_true, test_probs)}')
plt.plot([0, 1], [0, 1], 'k--')

idx = np.argmin(np.abs(thresholds - threshold))
plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Threshold ({threshold})')

plt.title('ROC Curve')
plt.legend()
plt.show()

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

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten() 

for i, err in enumerate(top_10_errors):
    img = tf.keras.utils.load_img(err['path'])
    axes[i].imshow(img)
    
    true_name = "FAKE" if err['true_label'] == 1 else "REAL"
    pred_name = "FAKE" if err['pred_label'] == 1 else "REAL"
    
    axes[i].set_title(f"True: {true_name}\nPred: {pred_name}\nProb: {err['pred_prob']:.2f}")
    axes[i].axis('off') 

plt.suptitle("10 Misclassified Test Images")
plt.tight_layout()
plt.show()
