"""
Generate training curve plot from the Lightning.ai run logs.
Data extracted from the terminal output (40 epochs, num_species=4, 21 classes).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ═══ Data from Lightning.ai training run (pasted from terminal) ═══
epochs = list(range(1, 41))

train_loss = [
    2.4870, 2.0033, 1.6414, 1.4295, 1.3048, 1.2138, 1.1373, 1.0788, 1.0274, 0.9894,
    0.9577, 0.9242, 0.8972, 0.8716, 0.8500, 0.8294, 0.8223, 0.8028, 0.7798, 0.7642,
    0.7540, 0.7399, 0.7210, 0.7133, 0.7492, 0.7333, 0.7155, 0.7015, 0.5825, 0.5465,
    0.5257, 0.5132, 0.5028, 0.4802, 0.4750, 0.4769, 0.4789, 0.4723, 0.4750, 0.4698,
]

train_acc = [
    11.73, 24.06, 36.69, 43.91, 48.48, 51.99, 55.34, 57.82, 60.04, 61.55,
    63.01, 64.42, 65.56, 66.60, 67.32, 68.22, 68.44, 69.37, 69.80, 70.50,
    70.80, 71.44, 72.14, 72.36, 71.26, 71.83, 72.35, 72.97, 77.84, 79.06,
    79.73, 80.29, 80.59, 81.43, 81.69, 81.92, 81.68, 81.92, 81.81, 81.94,
]

val_loss = [
    2.1387, 1.7280, 1.9335, 1.3093, 1.1981, 1.4302, 1.0600, 0.9666, 1.0686, 0.9671,
    0.8997, 0.8627, 0.9584, 0.9920, 0.9378, 0.8046, 1.8010, 1.5314, 0.7798, 0.7540,
    0.7399, 0.7210, 0.7133, 0.7492, 1.1873, 0.9369, 0.8357, 0.9889, 0.7362, 0.7451,
    0.7569, 0.7712, 0.7765, 0.7765, 0.7793, 0.7758, 0.7799, 0.7810, 0.7804, 0.7744,
]

val_acc = [
    18.69, 34.10, 31.26, 48.98, 53.27, 46.68, 57.45, 62.42, 59.75, 62.32,
    64.64, 66.67, 63.55, 62.27, 64.46, 68.48, 49.46, 56.40, 70.49, 71.20,
    70.48, 71.44, 72.14, 71.20, 59.49, 65.20, 67.94, 64.81, 71.81, 71.46,
    71.37, 71.33, 71.40, 71.64, 71.22, 71.54, 71.55, 71.14, 71.49, 71.57,
]

# ═══ Plot ═══
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=13)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 2.5)
ax1.grid(True, alpha=0.3)
ax1.axhline(np.log(21), color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax1.text(35, np.log(21)+0.05, 'log(21)', fontsize=9, color='gray')

# Accuracy
ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Accuracy')
ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training & Validation Accuracy', fontsize=13)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.axhline(100/21, color='gray', linestyle='--', alpha=0.5)
ax2.text(35, 100/21+1, 'Random (4.8%)', fontsize=9, color='gray')

# Annotate best val
best_epoch = np.argmax(val_acc) + 1
best_val = max(val_acc)
ax2.annotate(f'Best: {best_val:.1f}%\n(epoch {best_epoch})',
             xy=(best_epoch, best_val),
             xytext=(best_epoch+3, best_val-10),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

fig.suptitle('DeepSKAN Training — 4 Species, 21 Classes, 50k Samples, T4 GPU',
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('results/training_curve.png', dpi=200, bbox_inches='tight')
print("Saved: results/training_curve.png")
plt.close()
