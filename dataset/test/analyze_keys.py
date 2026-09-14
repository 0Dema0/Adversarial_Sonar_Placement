import numpy as np

p1 = "data/best_dataset_parts/best_sonar_split.npy"
p2 = "data/best_dataset_remaining_parts/best_sonar_split.npy"

a = np.load(p1, mmap_mode="r")
b = np.load(p2, mmap_mode="r")

print("Dataset 1:")
print("  dtype:", a.dtype)
print("  shape:", a.shape)
print("  values:", np.unique(a, return_counts=True))

print()
print("Dataset 2:")
print("  dtype:", b.dtype)
print("  shape:", b.shape)
print("  values:", np.unique(b, return_counts=True))