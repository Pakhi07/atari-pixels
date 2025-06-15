import numpy as np

data = np.load('data/actions/action_latent_pairs.npz')
# actions = data['actions']
# frames = data['frames']
# latents = data['latents']

# print("actions shape:", actions.shape)     # (N,)
# print("frames shape:", frames.shape)       # (N, 9, H, W)
# print("latents shape:", latents.shape)     # (N, latent_dim)


import matplotlib.pyplot as plt
import numpy as np

# Load data
frames = data['frames']
actions = data['actions']
latents = data['latents']

# Number of samples to plot
num_samples = 5
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

for i in range(num_samples):
    # Select i-th example from the dataset
    frame_set = frames[i + 70]       # (3, 3, 210, 160)
    action_set = actions[i + 70]     # (3, 1, 4)
    latent_vec = latents[i + 70]     # (35,)

    # Display 3 RGB frames
    for j in range(3):
        rgb = frame_set[j].transpose(1, 2, 0)     # (210, 160, 3)
        rgb = (rgb * 255).astype(np.uint8)        # scale to 0-255 if needed
        axes[i, j].imshow(rgb)
        if j == 1:
            action_str = np.array2string(action_set[j].squeeze(), precision=2, separator=',')
            axes[i, j].set_title(f'Action: {action_str}')
        elif j == 2:
            latent_str = np.array2string(latent_vec[:5], precision=2, separator=',')  # partial display
            axes[i, j].set_title(f'Latent[:5]: {latent_str}')
        else:
            axes[i, j].set_title(f'Sample {i} - Frame {j+1}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()