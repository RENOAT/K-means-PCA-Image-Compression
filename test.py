import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import io
import time
import os

# Load and preprocess the image
image_path = 'image/black_cat.png'  # Replace with your image path
original_image = io.imread(image_path) / 255.0  # Normalize to [0, 1]
if original_image.shape[2] == 4:
    original_image = original_image[:, :, :3]  # Remove alpha channel
print('Original Image Shape:', original_image.shape)
image_shape = original_image.shape
original_size = os.path.getsize(image_path)

# Function to calculate performance metrics
def calculate_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    psnr = 10 * np.log10(1 / mse)
    return mse, psnr

# Function to save compressed images
def save_image(image, filename):
    plt.imsave(filename, np.clip(image * 255, 0, 255).astype(np.uint8))

# K-Means Compression
k_values = [2, 5, 10, 20, 50, 100]
kmeans_mse, kmeans_psnr, kmeans_times, kmeans_ratios = [], [], [], []
kmeans_images = []

for k in k_values:
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(original_image.reshape(-1, 3))
    compressed_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image_shape)
    kmeans_images.append(compressed_image)
    mse, psnr = calculate_metrics(original_image, compressed_image)
    kmeans_mse.append(mse)
    kmeans_psnr.append(psnr)
    elapsed_time = time.time() - start_time
    kmeans_times.append(elapsed_time)
    filename = f'image/kmeans_compressed_k{k}.png'
    save_image(compressed_image, filename)
    compressed_size = os.path.getsize(filename)
    kmeans_ratios.append(original_size / compressed_size)

# PCA Compression
components_values = [2, 5, 10, 20, 50, 100]
pca_mse, pca_psnr, pca_times, pca_ratios, explained_variances = [], [], [], [], []
pca_images = []

for n_components in components_values:
    start_time = time.time()
    compressed_channels = []
    channel_variances = []
    for channel in range(3):
        flat_channel = original_image[:, :, channel]
        pca = PCA(n_components=min(n_components, flat_channel.shape[1]))
        transformed = pca.fit_transform(flat_channel)
        reconstructed_channel = pca.inverse_transform(transformed)
        compressed_channels.append(reconstructed_channel)
        channel_variances.append(np.sum(pca.explained_variance_ratio_))
    compressed_image = np.stack(compressed_channels, axis=2)
    pca_images.append(compressed_image)
    mse, psnr = calculate_metrics(original_image, compressed_image)
    pca_mse.append(mse)
    pca_psnr.append(psnr)
    elapsed_time = time.time() - start_time
    pca_times.append(elapsed_time)
    filename = f'image/pca_compressed_{n_components}.png'
    save_image(compressed_image, filename)
    compressed_size = os.path.getsize(filename)
    pca_ratios.append(original_size / compressed_size)
    explained_variances.append(np.mean(channel_variances))

# Plot K-Means and PCA compressed images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
k_titles = [f'K-Means k={k}' for k in k_values[:3]]
pca_titles = [f'PCA n={n}' for n in components_values[:3]]

for ax, img, title in zip(axes[0], kmeans_images[:3], k_titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

for ax, img, title in zip(axes[1], pca_images[:3], pca_titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(r'image/compressed_images_comparison_1.png')
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
k_titles = [f'K-Means k={k}' for k in k_values[3:]]
pca_titles = [f'PCA n={n}' for n in components_values[3:]]

for ax, img, title in zip(axes[0], kmeans_images[3:], k_titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

for ax, img, title in zip(axes[1], pca_images[3:], pca_titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(r'image/compressed_images_comparison_2.png')
plt.show()

# Plot MSE for K-Means and PCA
plt.figure(figsize=(10, 5))
plt.plot(k_values, kmeans_mse, label='K-Means MSE', marker='o')
plt.plot(components_values, pca_mse, label='PCA MSE', marker='o')
plt.xlabel('Number of Clusters / Components')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison of K-Means and PCA')
plt.legend()
plt.grid(True)
plt.savefig(r'image/mse_comparison.png')
plt.show()

# Plot PSNR for K-Means and PCA
plt.figure(figsize=(10, 5))
plt.plot(k_values, kmeans_psnr, label='K-Means PSNR', marker='o')
plt.plot(components_values, pca_psnr, label='PCA PSNR', marker='o')
plt.xlabel('Number of Clusters / Components')
plt.ylabel('Peak Signal-to-Noise Ratio (PSNR)')
plt.title('PSNR Comparison of K-Means and PCA')
plt.legend()
plt.grid(True)
plt.savefig(r'image/psnr_comparison.png')
plt.show()

# Plot Runtime Comparison
plt.figure(figsize=(10, 5))
plt.plot(k_values, kmeans_times, label='K-Means Runtime (s)', marker='o')
plt.plot(components_values, pca_times, label='PCA Runtime (s)', marker='o')
plt.xlabel('Number of Clusters / Components')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison of K-Means and PCA')
plt.legend()
plt.grid(True)
plt.savefig(r'image/runtime_comparison.png')
plt.show()

# Plot Compression Ratio Comparison
plt.figure(figsize=(10, 5))
plt.plot(k_values, kmeans_ratios, label='K-Means Compression Ratio', marker='o')
plt.plot(components_values, pca_ratios, label='PCA Compression Ratio', marker='o')
plt.xlabel('Number of Clusters / Components')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratio Comparison of K-Means and PCA')
plt.legend()
plt.grid(True)
plt.savefig(r'image/compression_ratio_comparison.png')
plt.show()

# Plot PCA Explained Variance
plt.figure(figsize=(10, 5))
plt.plot(components_values, explained_variances, label='Explained Variance Ratio', marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for PCA')
plt.grid(True)
plt.savefig(r'image/explained_variance_pca.png')
plt.show()
