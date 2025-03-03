import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from simplified_pinn import SimplifiedPINN, train_simplified_pinn

def visualize_temperature_field(model, domain_bounds, time_point=1.0, resolution=50, save_path=None):
    """Visualize temperature field predicted by the PINN model"""
    device = next(model.parameters()).device
    model.eval()
    
    # Create grid for xy plane at z=0 (surface)
    x = np.linspace(domain_bounds['x_min'], domain_bounds['x_max'], resolution)
    y = np.linspace(domain_bounds['y_min'], domain_bounds['y_max'], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # z=0 (surface)
    T = np.ones_like(X) * time_point
    
    # Reshape for batch prediction
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]).T
    
    # Predict temperature using model in batches to avoid memory issues
    batch_size = 1000
    num_batches = len(points) // batch_size + (1 if len(points) % batch_size != 0 else 0)
    temps_list = []
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            points_tensor = torch.tensor(batch_points, dtype=torch.float32).to(device)
            batch_temps = model(
                points_tensor[:, 0:1], 
                points_tensor[:, 1:2], 
                points_tensor[:, 2:3], 
                points_tensor[:, 3:4]
            )
            temps_list.append(batch_temps.cpu().numpy())
    
    temps = np.concatenate(temps_list).reshape(X.shape)
    
    # Plot temperature field
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, temps, 50, cmap='hot')
    plt.colorbar(contour, label='Temperature (K)')
    plt.title(f'Temperature Field at z=0, t={time_point}s')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    
    # Add laser path
    laser_params = model.laser_params
    start_x, start_y = laser_params['start_pos']
    end_x, end_y = laser_params['end_pos']
    plt.plot([start_x, end_x], [start_y, end_y], 'w--', linewidth=2, label='Laser path')
    
    # Mark current laser position if within deposition time
    if time_point <= 2.0:  # Deposition time from Table 2
        velocity = laser_params['velocity'] / 1000.0  # Convert mm/s to m/s
        path_length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        travel_time = path_length / velocity
        
        if time_point <= travel_time:
            fraction = time_point / travel_time
            current_x = start_x + (end_x - start_x) * fraction
            current_y = start_y + (end_y - start_y) * fraction
            plt.plot(current_x, current_y, 'wo', markersize=8, label='Current laser position')
    
    plt.legend(loc='upper right')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return temps

def main():
    """Main function to run the simplified PINN model for LMD"""
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('./figures', exist_ok=True)
    
    # Define domain bounds based on Table 2 in the paper (converted to meters)
    domain_bounds = {
        'x_min': 0.0, 'x_max': 0.04,  # 40mm length converted to m
        'y_min': 0.0, 'y_max': 0.02,  # 20mm width converted to m
        'z_min': 0.0, 'z_max': 0.005, # 5mm height converted to m
        't_min': 0.0, 't_max_deposition': 2.0, 't_max_cooling': 100.0
    }
    
    print("\n" + "="*50)
    print("SIMPLIFIED PINN FOR LASER METAL DEPOSITION")
    print("="*50)
    
    # Create and initialize PINN model
    print("Initializing simplified PINN model...")
    model = SimplifiedPINN(domain_bounds)
    print(f"Model created successfully")
    
    # Train the model with a focus on numerical stability
    print("\nTraining model...")
    model = train_simplified_pinn(
        model, 
        domain_bounds, 
        epochs=50,            # Start with fewer epochs for quick feedback
        lr=1e-3,              # Higher learning rate for faster learning
        points_per_batch=500, # Smaller batch size for better memory usage
        lambda_pde=1.0,       # Focus on PDE residual
        lambda_ic=0.1,        # Lower weight for initial conditions
        lambda_bc=0.1         # Lower weight for boundary conditions
    )
    
    # Visualize temperature field at different time points
    print("\nVisualizing temperature field...")
    for t in [0.5, 1.0, 2.0]:
        visualize_temperature_field(
            model, domain_bounds, time_point=t, resolution=50,
            save_path=f'./figures/simplified_temp_t{t}.png'
        )
    
    print("\nSimulation complete!")
    print("\nRecommended next steps:")
    print("1. If this run is successful, train for more epochs (100-200)")
    print("2. Adjust learning rate if necessary")
    print("3. After obtaining reasonable temperature fields, implement the cooling stage")

if __name__ == "__main__":
    main()
