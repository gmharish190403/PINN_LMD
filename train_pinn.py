import torch
import torch.optim as optim
from pinn_model import PINN, train_PINN, P_base, pde_residual, bc_residual, ic_residual, output_transform, input_transform, thermal_conductivity, heat_source
from data_preparation import x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, N_f, N_b, N_i
import os

# ---------------------------- Device Setup ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------- Training Setup ----------------------------
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate for stability
epochs = 20000
lr_schedule = {5000, 10000, 15000}  # Decay learning rate at these epochs
save_path = "pinn_model.pth"

# Constants for scaling (ensure these match pinn_model.py)
rho = 7780.0    # kg/m³
cp = 600.0      # J/kg·K
h = 20.0        # W/m²·K
T_inf = 293.15  # K
epsilon = 0.7   # Emissivity
sigma = 5.67e-8 # W/m²·K⁴
T_range = 3000.0  # Temperature range
k_max = 314.2   # Max thermal conductivity (W/m·K)
P_scale = 2000.0 * 0.75  # Max heat source scale (W)

# Scaled parameters
rho_scaled = rho / 7780.0
cp_scaled = cp / 600.0
h_scaled = h / 20.0

# Domain bounds (in meters, seconds)
L_x, L_y, L_z = 0.04, 0.02, 0.005  # m
t_total = 102.0  # s
x_min, x_max = 0.0, L_x
y_min, y_max = 0.0, L_y
z_min, z_max = 0.0, L_z
t_min, t_max = 0.0, t_total

# Debug: Check input shapes
print(f"x_f shape: {x_f.shape}, y_f shape: {y_f.shape}, z_f shape: {z_f.shape}, t_f shape: {t_f.shape}")
print(f"x_b shape: {x_b.shape}, y_b shape: {y_b.shape}, z_b shape: {z_b.shape}, t_b shape: {t_b.shape}")
print(f"x_i shape: {x_i.shape}, y_i shape: {y_i.shape}, z_i shape: {z_i.shape}, t_i shape: {t_i.shape}")

# Load existing model for transfer learning
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("✅ Loaded pre-trained model for transfer learning")

# Train the model with debugging
def train_PINN_debug(model, optimizer, epochs, P, save_path, x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, lr_schedule, N_f, N_b, N_i):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss
        f_pde = pde_residual(model, x_f, y_f, z_f, t_f, P)
        if torch.isnan(f_pde).any() or torch.isinf(f_pde).any():
            print("NaN or Inf in PDE residual at epoch", epoch)
            break
        loss_pde = torch.mean(f_pde ** 2) / N_f
        
        # Boundary condition loss
        f_bc = bc_residual(model, x_b, y_b, z_b, t_b, P)
        if torch.isnan(f_bc).any() or torch.isinf(f_bc).any():
            print("NaN or Inf in BC residual at epoch", epoch)
            break
        loss_bc = torch.mean(f_bc ** 2) / (6 * N_b)
        
        # Initial condition loss
        f_ic = ic_residual(model, x_i, y_i, z_i, t_i)
        if torch.isnan(f_ic).any() or torch.isinf(f_ic).any():
            print("NaN or Inf in IC residual at epoch", epoch)
            break
        loss_ic = torch.mean(f_ic ** 2) / N_i
        
        # Total loss with adjusted weights
        loss = loss_pde + 1e-4 * loss_bc + 1e-4 * loss_ic  # Match paper’s weights
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf in total loss at epoch", epoch)
            break
        
        loss.backward()
        optimizer.step()
        
        if epoch in lr_schedule:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5

        if epoch % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}, IC Loss: {loss_ic.item():.6f}")

    torch.save(model.state_dict(), save_path)

# Train the model
print("Starting training...")
train_PINN_debug(model, optimizer, epochs, P_base, save_path, x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, lr_schedule, N_f, N_b, N_i)
print("✅ Model saved successfully!")
