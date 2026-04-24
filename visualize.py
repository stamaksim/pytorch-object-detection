import matplotlib.pyplot as plt

train_loss = [
0.0198, 0.0186, 0.0189, 0.0186, 0.0167, 0.0151, 0.0151,
0.0150, 0.0132, 0.0118, 0.0138, 0.0134, 0.0133,
0.0140, 0.0134, 0.0123, 0.0118
]

val_loss = [
0.0294, 0.0309, 0.0339, 0.0299, 0.0316, 0.0340, 0.0320,
0.0284, 0.0276, 0.0271, 0.0326, 0.0266, 0.0322,
0.0289, 0.0276, 0.0282, 0.0267
]

plt.figure(figsize=(10,6))
plt.plot(train_loss, marker='o', label='Training Loss')
plt.plot(val_loss, marker='o', label='Validation Loss')

plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("learning_curve.png")
plt.show()
