from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    print(f"Saving {len(frames)} frames as gif")
    #Mess with this to change frame size
    dpi=50
    plt.figure(figsize=(frames[0].shape[1] / float(dpi), frames[0].shape[0] / float(dpi)), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)
    anim.save(path + filename, writer='imagemagick', fps=30)

def calculate_rewards(observation):
    # Unpack observation values
    player1_pos = np.array([observation[0], observation[1]])
    player1_angle = observation[2]
    player1_vel = np.array([observation[3], observation[4]])
    player1_angular_vel = observation[5]

    player2_pos = np.array([observation[6], observation[7]])
    player2_angle = observation[8]
    player2_vel = np.array([observation[9], observation[10]])
    player2_angular_vel = observation[11]

    puck_pos = np.array([observation[12], observation[13]])
    puck_vel = np.array([observation[14], observation[15]])

    puck_possession_time_player1 = observation[16]
    puck_possession_time_player2 = observation[17]

    # Initialize reward
    reward = 0

    # Reward for puck possession time
    reward += puck_possession_time_player1 - puck_possession_time_player2

    # Reward for puck direction towards opponent's goal
    if puck_vel[0] > 0:
        reward += 1

    # Reward for puck being in the opponent's half
    if puck_pos[0] > 0:
        reward += 1

    # Negative reward for distance to the puck
    reward -= np.linalg.norm(player1_pos - puck_pos)

    # Reward for agent speed towards puck
    vec_to_puck = puck_pos - player1_pos
    if np.dot(player1_vel, vec_to_puck) < 0:
        reward += 1

    # Negative reward for high player speed (energy conservation)
    reward -= np.linalg.norm(player1_vel)

    # Reward for facing towards the puck
    direction_to_puck = np.arctan2(vec_to_puck[1], vec_to_puck[0])
    angle_diff = player1_angle - direction_to_puck
    reward += np.cos(angle_diff)

    return reward