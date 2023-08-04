from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch 
import random


def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):
    print(f"Saving {len(frames)} frames as gif")
    # Mess with this to change frame size
    dpi = 50
    plt.figure(
        figsize=(frames[0].shape[1] / float(dpi), frames[0].shape[0] / float(dpi)),
        dpi=dpi,
    )

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=10)
    anim.save(path + filename, writer="imagemagick", fps=30)
