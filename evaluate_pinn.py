import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from pinn_model import PINN, input_transform, output_transform, device

# ---------------------------- Model Evaluation Function ----------------------------
def evaluate_model(model, x, y, z, t):
    """Evaluates the trained PINN model at given (x, y, z, t) points."""
    model.eval()
    x, y, z, t = input_transform(x, y, z, t)
    x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)
    
    with torch.no_grad():
        temp_scaled = model(x, y, z, t)
        temperature = output_transform(temp_scaled)
    return temperature

# ---------------------------- Plotting Functions ----------------------------
def plot_1d_profiles(model, x_min, x_max, y_center, z_center, t_values, save_path='temperature_profiles.png'):
    """Plot temperature profiles along x-axis at fixed y, z, and multiple times."""
    x_points = torch.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_points = torch.ones_like(x_points) * y_center
    z_points = torch.ones_like(x_points) * z_center
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'red']
    labels = [f't={t:.1f}s (PINN)' for t in t_values]
    
    for t, color, label in zip(t_values, colors, labels):
        t_points = torch.ones_like(x_points) * t
        temps = evaluate_model(model, x_points, y_points, z_points, t_points)
        x_mm = x_points.cpu().numpy() * 1000  # Convert to mm
        temps_k = temps.cpu().numpy()
        plt.plot(x_mm, temps_k, color=color, label=label)
    
    plt.xlabel('x (mm)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Profiles Along x-Axis at Different Times (PINN)')
    plt.legend()
    plt.grid(True)
    plt.ylim(293, 3000)  # Adjusted to match paper’s range (ambient to max temp)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ 1D profiles saved as '{save_path}'")

def plot_2d_contour(model, x_min, x_max, z_min, z_max, y_center, t_value, save_path='temperature_contour.png'):
    """Plot 2D temperature contour in x-z plane at fixed y and t."""
    x_points = torch.linspace(x_min, x_max, 50)
    z_points = torch.linspace(z_min, z_max, 50)
    X, Z = torch.meshgrid(x_points, z_points, indexing='ij')
    y_points = torch.ones_like(X) * y_center
    t_points = torch.ones_like(X) * t_value
    
    temps = evaluate_model(model, X.flatten().reshape(-1, 1), 
                          y_points.flatten().reshape(-1, 1), 
                          Z.flatten().reshape(-1, 1), 
                          t_points.flatten().reshape(-1, 1))
    temps = temps.reshape(X.shape).cpu().numpy()
    X_mm = X.cpu().numpy() * 1000  # Convert to mm
    Z_mm = Z.cpu().numpy() * 1000
    
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X_mm, Z_mm, temps, levels=20, cmap='hot')
    plt.colorbar(contour, label='Temperature (K)')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.title(f'Temperature Contour in x-z Plane at y={y_center*1000:.0f} mm, t={t_value:.1f} s')
    plt.savefig(save_path)
    plt.show()
    print(f"✅ 2D contour saved as '{save_path}'")

# ---------------------------- Main Evaluation ----------------------------
if __name__ == "__main__":
    # Load trained model
    model = PINN().to(device)
    save_path_base = "pinn_base.pth"
    save_path_new = "pinn_new.pth"
    
    # Try loading final (transfer-learned) model first, then base model
    for path in [save_path_new, save_path_base]:
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f"✅ Model loaded successfully from '{path}'!")
            break
        except FileNotFoundError:
            print(f"⚠️ Model file '{path}' not found.")
        except Exception as e:
            print(f"⚠️ Error loading model from '{path}': {e}")
    else:
        raise FileNotFoundError("No trained model found. Run train_pinn.py first.")

    # Domain bounds (consistent with data_preparation.py and pinn_model.py)
    x_min, x_max = 0.0, 0.04  # 0–40 mm
    y_min, y_max = 0.0, 0.02  # 0–20 mm
    z_min, z_max = 0.0, 0.005  # 0–5 mm
    y_center = 0.01  # 10 mm (laser path center)
    z_center = 0.0025  # 2.5 mm (mid-depth)
    t_values = [0.5, 1.0, 2.0]  # Times during deposition

    # 1D Temperature Profiles
    plot_1d_profiles(model, x_min, x_max, y_center, z_center, t_values)

    # 2D Temperature Contour at t = 2.0 s
    plot_2d_contour(model, x_min, x_max, z_min, z_max, y_center, 2.0)

    # Validation point (max temp from paper, Fig. 5)
    test_in = torch.tensor([[0.02, 0.01, 0.0025, 2.0]], device=device)  # Center at t=2s
    test_out = torch.tensor([[2766.0]], device=device)  # Approx max temp
    pred_temp = evaluate_model(model, test_in[:, 0:1], test_in[:, 1:2], 
                              test_in[:, 2:3], test_in[:, 3:4])
    error = (pred_temp - test_out).abs().item()
    print(f"Validation at (x=20 mm, y=10 mm, z=2.5 mm, t=2.0 s):")
    print(f"Predicted: {pred_temp.item():.2f} K, Expected: {test_out.item():.2f} K, Error: {error:.2f} K")
