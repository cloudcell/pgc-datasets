import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Load image
img = Image.open("./research_records/Figure_1.png").convert("RGB")

# --- DEBUG: Skip cropping, just resize full image ---
cropped = img.resize((512, 512))
cropped.save("_debug_cropped_full.png")
cropped.show()
data = np.array(cropped) / 255.0
if data.ndim == 2:
    data = np.stack([data]*3, axis=-1)
if data.shape[2] == 4:
    data = data[..., :3]
data = np.transpose(data, (1, 0, 2))
print(f"[DEBUG] data shape: {data.shape}, min: {data.min()}, max: {data.max()}")

# Create cylindrical surface
h, w, _ = data.shape
print(f"[DEBUG] meshgrid h: {h}, w: {w}")
theta = np.linspace(0, 2 * np.pi, h)
z = np.linspace(0, 1, w)
theta, z = np.meshgrid(theta, z)
x = np.cos(theta)
y = np.sin(theta)
print(f"[DEBUG] x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")

# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
try:
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=data, shade=False, edgecolor='k', linewidth=0.1, antialiased=True)
except Exception as e:
    print(f"[ERROR] plot_surface failed: {e}")
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(z / z.max()), shade=False)
ax.set_axis_off()
ax.set_title("Spectrogram Wrapped Lengthwise on Tube (No Cropping)")
plt.tight_layout()
plt.savefig("tube_output.png", dpi=300, bbox_inches='tight')
# plt.show()  # Commented out to avoid opening window
print("[INFO] Tube visualization saved to tube_output.png")
