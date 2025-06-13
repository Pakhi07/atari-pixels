import numpy as np

data = np.load('data/actions/action_latent_pairs.npz')
actions = data['actions']
frames = data['frames']
latents = data['latents']

print("actions shape:", actions.shape)     # (N,)
print("frames shape:", frames.shape)       # (N, 9, H, W)
print("latents shape:", latents.shape)     # (N, latent_dim)


import matplotlib.pyplot as plt
import numpy as np

# Load data
frames = data['frames']
actions = data['actions']
latents = data['latents']

# Number of samples to plot
num_samples = 5

# Create figure with one row per sample, and 3 columns for RGB frames
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

for i in range(num_samples):
    # Extract 3 RGB frames from the stacked 9-channel input
    rgb1 = frames[i+70][0:3].transpose(1, 2, 0)
    rgb2 = frames[i+70][3:6].transpose(1, 2, 0)
    rgb3 = frames[i+70][6:9].transpose(1, 2, 0)

    # Scale from [0,1] to [0,255] if needed
    rgb1 = (rgb1 * 255).astype(np.uint8)
    rgb2 = (rgb2 * 255).astype(np.uint8)
    rgb3 = (rgb3 * 255).astype(np.uint8)

    # Plot each frame
    axes[i, 0].imshow(rgb1)
    axes[i, 0].set_title(f'Sample {i} - Frame 1')

    axes[i, 1].imshow(rgb2)
    axes[i, 1].set_title(f'Action: {actions[i]}')

    axes[i, 2].imshow(rgb3)
    axes[i, 2].set_title(f'Latent')

    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()
plt.show()
