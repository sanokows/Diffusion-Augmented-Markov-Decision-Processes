"""
Utility functions for saving and plotting end-effector trajectories.
"""

import numpy as np
from pathlib import Path


def save_and_plot_trajectories(
    all_trajectories: list,
    save_dir: str,
    env_name: str,
    episode_info: dict = None,
    plot_trajectories: bool = True,
):
    """
    Save trajectory data and optionally create visualization plots using seaborn for professional styling.
    
    Args:
        all_trajectories: List of trajectory dictionaries from multiple episodes
        save_dir: Directory to save trajectory data and plots
        env_name: Name of the environment
        episode_info: Optional dictionary with episode statistics (returns, lengths, success rates)
        plot_trajectories: Whether to generate trajectory plots (default: True)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    
    # Set seaborn style for better-looking plots
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw trajectory data
    trajectory_file = save_path / f"{env_name}_trajectories.npz"
    np.savez(trajectory_file, trajectories=all_trajectories, episode_info=episode_info)
    print(f"Saved trajectory data to {trajectory_file}")
    
    # Early return if plotting is disabled
    if not plot_trajectories:
        print("Skipping trajectory plotting (plot_trajectories=False)")
        return trajectory_file, None
    
    # Create visualizations
    num_episodes = len(all_trajectories)
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", num_episodes)
    
    # Plot 1: 3D trajectory visualization
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'End-Effector Trajectory Analysis: {env_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 3D plot with seaborn styling
    ax1 = fig.add_subplot(221, projection='3d')
    
    for ep_idx, traj in enumerate(all_trajectories):
        if not traj:
            continue
        ee_positions = np.array([t['ee_pos'] for t in traj])  # Shape: (steps, num_envs, 3)
        
        # Plot trajectory for first environment in each episode
        if ee_positions.shape[1] > 0:
            traj_data = ee_positions[:, 0, :]  # First environment
            ax1.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], 
                    color=colors[ep_idx], alpha=0.7, linewidth=2.5,
                    label=f'Episode {ep_idx + 1}')
            # Mark start and end points
            ax1.scatter(traj_data[0, 0], traj_data[0, 1], traj_data[0, 2], 
                       color=colors[ep_idx], marker='o', s=150, edgecolors='black', 
                       linewidths=2, alpha=0.9, zorder=10)
            ax1.scatter(traj_data[-1, 0], traj_data[-1, 1], traj_data[-1, 2], 
                       color=colors[ep_idx], marker='*', s=250, edgecolors='black', 
                       linewidths=2, alpha=0.9, zorder=10)
    
    ax1.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z Position (m)', fontsize=11, fontweight='bold')
    ax1.set_title('3D Trajectories', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: XY plane projection with seaborn style
    ax2 = fig.add_subplot(222)
    for ep_idx, traj in enumerate(all_trajectories):
        if not traj:
            continue
        ee_positions = np.array([t['ee_pos'] for t in traj])
        if ee_positions.shape[1] > 0:
            traj_data = ee_positions[:, 0, :]
            ax2.plot(traj_data[:, 0], traj_data[:, 1], 
                    color=colors[ep_idx], alpha=0.7, linewidth=2.5,
                    label=f'Episode {ep_idx + 1}')
            ax2.scatter(traj_data[0, 0], traj_data[0, 1], 
                       color=colors[ep_idx], marker='o', s=150, edgecolors='black', 
                       linewidths=2, alpha=0.9, zorder=10)
            ax2.scatter(traj_data[-1, 0], traj_data[-1, 1], 
                       color=colors[ep_idx], marker='*', s=250, edgecolors='black', 
                       linewidths=2, alpha=0.9, zorder=10)
    
    ax2.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax2.set_title('XY Plane Projection', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='black')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    sns.despine(ax=ax2, offset=5)
    
    # Plot 3: Position over time with seaborn styling
    ax3 = fig.add_subplot(223)
    
    # Prepare data for seaborn-style plotting
    for ep_idx, traj in enumerate(all_trajectories):
        if not traj:
            continue
        ee_positions = np.array([t['ee_pos'] for t in traj])
        if ee_positions.shape[1] > 0:
            traj_data = ee_positions[:, 0, :]
            steps = np.arange(len(traj_data))
            
            # Plot with different line styles for X, Y, Z
            ax3.plot(steps, traj_data[:, 0], color=colors[ep_idx], alpha=0.8, 
                    linestyle='-', linewidth=2, label=f'Ep {ep_idx + 1} (X)')
            ax3.plot(steps, traj_data[:, 1], color=colors[ep_idx], alpha=0.6, 
                    linestyle='--', linewidth=1.5, label=f'Ep {ep_idx + 1} (Y)')
            ax3.plot(steps, traj_data[:, 2], color=colors[ep_idx], alpha=0.4, 
                    linestyle=':', linewidth=1.5, label=f'Ep {ep_idx + 1} (Z)')
    
    ax3.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Position (m)', fontsize=11, fontweight='bold')
    ax3.set_title('Position Components over Time', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=7, ncol=2, framealpha=0.9, edgecolor='black')
    ax3.grid(True, alpha=0.3)
    sns.despine(ax=ax3, offset=5)
    
    # Plot 4: Episode statistics with seaborn styling
    ax4 = fig.add_subplot(224)
    if episode_info:
        returns = episode_info.get('returns', [])
        successes = episode_info.get('successes', [])
        
        if returns:
            episodes = np.arange(1, len(returns) + 1)
            
            # Use seaborn barplot styling
            ax4_twin = ax4.twinx()
            
            # Plot returns with seaborn color
            bars1 = ax4.bar(episodes - 0.2, returns, 0.4, 
                          color=sns.color_palette("deep")[0], 
                          alpha=0.8, label='Return', edgecolor='black', linewidth=1)
            ax4.set_xlabel('Episode', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Return', fontsize=11, fontweight='bold', 
                          color=sns.color_palette("deep")[0])
            ax4.tick_params(axis='y', labelcolor=sns.color_palette("deep")[0], labelsize=10)
            
            # Plot success rates
            if successes:
                bars2 = ax4_twin.bar(episodes + 0.2, successes, 0.4, 
                                    color=sns.color_palette("deep")[2], 
                                    alpha=0.8, label='Success', edgecolor='black', linewidth=1)
                ax4_twin.set_ylabel('Success Rate', fontsize=11, fontweight='bold',
                                  color=sns.color_palette("deep")[2])
                ax4_twin.tick_params(axis='y', labelcolor=sns.color_palette("deep")[2], labelsize=10)
                ax4_twin.set_ylim([0, 1.1])
            
            ax4.set_title('Episode Performance', fontsize=12, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper left', fontsize=9, framealpha=0.9, edgecolor='black')
            sns.despine(ax=ax4, offset=5, right=False)
    else:
        ax4.text(0.5, 0.5, 'No episode statistics available', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, style='italic')
        ax4.set_title('Episode Statistics', fontsize=12, fontweight='bold', pad=10)
        sns.despine(ax=ax4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_file = save_path / f"{env_name}_trajectory_plot.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved trajectory plot to {plot_file}")
    
    # Create individual episode plots for detailed analysis with seaborn styling
    for ep_idx, traj in enumerate(all_trajectories):
        if not traj:
            continue
        
        fig2 = plt.figure(figsize=(14, 10))
        fig2.suptitle(f'Episode {ep_idx + 1} - Detailed Analysis', 
                     fontsize=14, fontweight='bold', y=0.98)
        ee_positions = np.array([t['ee_pos'] for t in traj])
        
        if ee_positions.shape[1] > 0:
            traj_data = ee_positions[:, 0, :]
            steps = np.arange(len(traj_data))
            
            # Use seaborn color palette for consistency
            ep_colors = sns.color_palette("Set2", 8)
            
            # 3D trajectory with seaborn styling
            ax = fig2.add_subplot(221, projection='3d')
            ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], 
                   color=ep_colors[0], linewidth=2.5, alpha=0.8)
            ax.scatter(traj_data[0, 0], traj_data[0, 1], traj_data[0, 2], 
                      color=ep_colors[2], marker='o', s=150, label='Start',
                      edgecolors='black', linewidths=1.5, alpha=0.9)
            ax.scatter(traj_data[-1, 0], traj_data[-1, 1], traj_data[-1, 2], 
                      color=ep_colors[3], marker='*', s=250, label='End',
                      edgecolors='black', linewidths=1.5, alpha=0.9)
            ax.set_xlabel('X Position (m)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y Position (m)', fontsize=10, fontweight='bold')
            ax.set_zlabel('Z Position (m)', fontsize=10, fontweight='bold')
            ax.set_title('3D Trajectory', fontsize=11, fontweight='bold', pad=10)
            ax.legend(fontsize=9, framealpha=0.9, edgecolor='black')
            ax.grid(True, alpha=0.3)
            
            # X, Y, Z over time with seaborn styling
            ax2 = fig2.add_subplot(222)
            ax2.plot(steps, traj_data[:, 0], color=ep_colors[4], 
                    linewidth=2, alpha=0.8, label='X')
            ax2.plot(steps, traj_data[:, 1], color=ep_colors[5], 
                    linewidth=2, alpha=0.8, label='Y')
            ax2.plot(steps, traj_data[:, 2], color=ep_colors[6], 
                    linewidth=2, alpha=0.8, label='Z')
            ax2.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Position (m)', fontsize=10, fontweight='bold')
            ax2.set_title('Position Components', fontsize=11, fontweight='bold', pad=10)
            ax2.legend(fontsize=9, framealpha=0.9, edgecolor='black', loc='best')
            ax2.grid(True, alpha=0.3, axis='y')
            sns.despine(ax=ax2, offset=5)
            
            # Velocity (approximate) with seaborn styling
            ax3 = fig2.add_subplot(223)
            velocity = np.diff(traj_data, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            ax3.plot(steps[:-1], speed, color=ep_colors[7], linewidth=2.5, alpha=0.8)
            ax3.fill_between(steps[:-1], speed, alpha=0.3, color=ep_colors[7])
            ax3.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Speed (m/step)', fontsize=10, fontweight='bold')
            ax3.set_title('End-Effector Speed', fontsize=11, fontweight='bold', pad=10)
            ax3.grid(True, alpha=0.3, axis='y')
            sns.despine(ax=ax3, offset=5)
            
            # Rewards over time with seaborn styling
            ax4 = fig2.add_subplot(224)
            rewards = np.array([t['rewards'][0] for t in traj])  # First env
            total_reward = rewards.sum()
            
            # Create color gradient based on reward values
            reward_colors = rewards.copy()
            ax4.plot(steps, rewards, color=ep_colors[1], linewidth=2.5, alpha=0.8)
            ax4.fill_between(steps, rewards, alpha=0.3, color=ep_colors[1])
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
            ax4.set_xlabel('Time Step', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Reward', fontsize=10, fontweight='bold')
            ax4.set_title(f'Rewards (Total: {total_reward:.2f})', 
                         fontsize=11, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, axis='y')
            sns.despine(ax=ax4, offset=5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        episode_plot_file = save_path / f"{env_name}_episode_{ep_idx + 1}_trajectory.png"
        plt.savefig(episode_plot_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"Saved {num_episodes} individual episode plots with seaborn styling")
    
    return trajectory_file, plot_file
