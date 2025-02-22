import torch
import torch.nn as nn
import numpy as np

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- PINN Model ----------------------------
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 50),  # Input: (x, y, z, t) normalized to [-1, 1]
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)   # Output: Scaled temperature
        )
    
    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)  # Ensure correct format
        return self.net(inputs)

# ---------------------------- Constants & Material Properties ----------------------------
rho = 7780.0    # Density (kg/m³)
cp = 600.0      # Specific heat (J/kg·K)
h = 20.0        # Convection coefficient (W/m²·K)
T_inf = 293.15  # Ambient temperature (K)
epsilon = 0.7   # Emissivity (steel)
sigma = 5.67e-8 # Stefan-Boltzmann constant (W/m²·K⁴)
T_range = 3000.0  # Temperature range for output scaling
T_liquidus = 1690.0  # Liquidus temperature (K)
T_solidus = 1730.0   # Solidus temperature (K)

# Laser Parameters
P_base = 2000.0  # Initial Laser Power (W) for basic training
P_new = 2400.0   # New Laser Power for Transfer Learning (W)
eta = 0.75       # Absorption coefficient
v = 8.0          # Scanning speed (m/s, converted from 8000 mm/s)

# Scaled physical parameters
rho_scaled = rho / 7780.0  # 1.0
cp_scaled = cp / 600.0     # 1.0
h_scaled = h / 20.0        # 1.0
k_max = 314.2             # Max thermal conductivity (W/m·K)
P_scale = 2000.0 * 0.75   # Max heat source scale (W)

# Substrate size & Time Normalization (physical units in m and s)
L_x, L_y, L_z = 0.04, 0.02, 0.005  # m
t_total = 102.0  # Total time (s)

# Laser spot radius (m)
r_x, r_y, r_z = 0.003, 0.003, 0.001  # m

# Domain bounds for normalization
x_min, x_max = 0.0, L_x  # m
y_min, y_max = 0.0, L_y  # m
z_min, z_max = 0.0, L_z  # m
t_min, t_max = 0.0, t_total  # s

# ---------------------------- Transform Functions ----------------------------
def input_transform(x, y, z, t):
    x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    z = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
    t = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    return x, y, z, t

def output_transform(u_scaled):
    return torch.clamp(T_range * nn.Softplus()(u_scaled) + T_inf, min=T_inf, max=T_inf + T_range)

# ---------------------------- Thermal Conductivity ----------------------------
def thermal_conductivity(T):
    T_np = np.clip(T.detach().cpu().numpy(), T_inf, T_inf + T_range)
    k = np.zeros_like(T_np)
    
    mask_low = T_np <= 1773.15
    mask_high = T_np > 1773.15
    
    k[mask_low] = 1.04 * np.exp(-T_np[mask_low] / 2) - 0.34267 * T_np[mask_low] + 314.2
    k[mask_high] = -53.704 * T_np[mask_high]**2 - 3.267 * np.exp(-T_np[mask_high]) + 4 * np.exp(2)
    
    k[np.isnan(k)] = 0.0  # Replace NaNs with 0
    k[np.isinf(k)] = 314.2  # Replace Infs with max value
    
    return torch.tensor(k / k_max, device=device, requires_grad=False)  # Scale k

# ---------------------------- Heat Source (Laser) ----------------------------
def heat_source(x, y, z, t, P):
    x_phys = (x + 1.0) * L_x / 2  # [-1, 1] -> [0, 0.04] m
    y_phys = (y + 1.0) * L_y / 2  # [-1, 1] -> [0, 0.02] m
    z_phys = (z + 1.0) * L_z / 2  # [-1, 1] -> [0, 0.005] m
    t_phys = (t + 1.0) * t_total / 2  # [-1, 1] -> [0, 102] s
    
    x_laser = torch.where(t_phys <= 2.0, v * t_phys, torch.zeros_like(t_phys))
    r2_x = (x_phys - x_laser) ** 2 / (r_x ** 2)
    r2_y = y_phys ** 2 / (r_y ** 2)
    r2_z = z_phys ** 2 / (r_z ** 2)
    
    Q = eta * (P / P_scale) * torch.exp(-(r2_x + r2_y + r2_z)) * (t_phys <= 2.0)
    return Q / (L_x * L_y * L_z) / T_range  # Normalize by volume and T_range

# ---------------------------- Physics-Informed Loss Functions ----------------------------
def pde_residual(model, x, y, z, t, P):
    x, y, z, t = input_transform(x, y, z, t)
    x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)

    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)

    u_scaled = model(x, y, z, t)
    u = output_transform(u_scaled)  # Real temperature (K)

    # Compute derivatives with additional normalization for time derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (t_max - t_min) * 2 / (T_range * (t_max - t_min))
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (x_max - x_min) * 2 / T_range
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] / ((x_max - x_min) * 2) ** 2 / T_range
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (y_max - y_min) * 2 / T_range
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0] / ((y_max - y_min) * 2) ** 2 / T_range
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (z_max - z_min) * 2 / T_range
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0] / ((z_max - z_min) * 2) ** 2 / T_range

    k = thermal_conductivity(u)  # Scaled by k_max
    Q = heat_source(x, y, z, t, P)  # Scaled by P_scale, volume, and T_range
    latent_heat = 250000.0  # J/kg (typical for steel, adjust from paper)
    phase_factor = torch.where((u >= T_liquidus) & (u <= T_solidus), latent_heat * rho_scaled, 0.0) / (T_range * (t_max - t_min) * 2)
    u_t += phase_factor  # Adjust time derivative for latent heat

    # Re-scaled heat equation residual with additional normalization
    f = (rho_scaled * cp_scaled * u_t - k * (u_xx + u_yy + u_zz) - Q)  # Already normalized by T_range in derivatives
    if torch.isnan(f).any() or torch.isinf(f).any():
        print("NaN or Inf in PDE residual")
    return f

def bc_residual(model, x, y, z, t, P):
    x, y, z, t = input_transform(x, y, z, t)
    x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)

    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)

    u_scaled = model(x, y, z, t)
    u = output_transform(u_scaled)  # Real temperature (K)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (x_max - x_min) * 2 / T_range
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (y_max - y_min) * 2 / T_range
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0] / (z_max - z_min) * 2 / T_range

    k = thermal_conductivity(u)  # Scaled by k_max
    Q = heat_source(x, y, z, t, P) * (z == 1.0)  # Laser heat only on top

    # Re-scaled boundary conditions with consistent normalization
    bc_x_neg = (-k * u_x - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2))
    bc_x_pos = (k * u_x - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2))
    bc_y_neg = (-k * u_y - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2))
    bc_y_pos = (k * u_y - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2))
    bc_z_neg = (-k * u_z - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2))
    bc_z_pos = (-k * u_z - h_scaled * (u - T_inf) / T_range - epsilon * sigma * (u**4 - T_inf**4) / (h_scaled * T_range**2) + Q * (t <= (2.0 / t_total) * 2 - 1.0))

    f_bc = torch.cat([bc_x_neg, bc_x_pos, bc_y_neg, bc_y_pos, bc_z_neg, bc_z_pos], dim=0)
    if torch.isnan(f_bc).any() or torch.isinf(f_bc).any():
        print("NaN or Inf in BC residual")
    return f_bc

def ic_residual(model, x, y, z, t):
    x, y, z, t = input_transform(x, y, z, t)
    x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)

    u_scaled = model(x, y, z, t)
    u = output_transform(u_scaled)
    f_ic = (u - T_inf) / T_range
    if torch.isnan(f_ic).any() or torch.isinf(f_ic).any():
        print("NaN or Inf in IC residual")
    return f_ic

# ---------------------------- Training Function ----------------------------
def train_PINN(model, optimizer, epochs, P, save_path, x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, lr_schedule, N_f, N_b, N_i):
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        f_pde = pde_residual(model, x_f, y_f, z_f, t_f, P)
        loss_pde = torch.mean(f_pde ** 2) / N_f
        
        f_bc = bc_residual(model, x_b, y_b, z_b, t_b, P)
        loss_bc = torch.mean(f_bc ** 2) / (6 * N_b)
        
        f_ic = ic_residual(model, x_i, y_i, z_i, t_i)
        loss_ic = torch.mean(f_ic ** 2) / N_i
        
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

    torch.save(model.state_dict(), save_path, weights_only=True)  # Use weights_only=True for safety

# Export variables for use in train_pinn.py
if __name__ == "__main__":
    globals().update({
        'PINN': PINN,
        'train_PINN': train_PINN,
        'P_base': P_base,
        'pde_residual': pde_residual,
        'bc_residual': bc_residual,
        'ic_residual': ic_residual,
        'output_transform': output_transform,
        'input_transform': input_transform,
        'thermal_conductivity': thermal_conductivity,
        'heat_source': heat_source
    })
