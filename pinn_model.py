import torch
import torch.nn as nn

# Physical constants directly defined
x_min, x_max = 0.0, 0.04  # m
y_min, y_max = 0.0, 0.02  # m
z_min, z_max = 0.0, 0.005  # m
t_min, t_max = 0.0, 102.0  # s
L_x, L_y, L_z = x_max - x_min, y_max - y_min, z_max - z_min  # Domain lengths
t_total = t_max - t_min  # Total time

T_inf = 293.15  # K (ambient temperature)
T_range = 3000.0  # K (temperature range for normalization)
T_liquidus = 1673.15  # K (liquidus temperature for Ti-6Al-4V)
T_solidus = 1873.15  # K (solidus temperature for Ti-6Al-4V)
rho = 4420.0  # kg/m^3 (density)
cp = 526.3  # J/(kg·K) (specific heat capacity)
k_max = 31.4  # W/(m·K) (maximum thermal conductivity)
h = 20.0  # W/(m^2·K) (convective heat transfer coefficient)
epsilon = 0.3  # Emissivity
sigma = 5.6704e-8  # W/(m^2·K^4) (Stefan-Boltzmann constant)
r_x, r_y, r_z = 0.003, 0.003, 0.001  # m (Gaussian beam radii)
eta = 0.4  # Laser absorption efficiency
v = 0.01  # m/s (laser scanning speed)
P_base = 2000.0  # W (base laser power)
P_new = 2400.0  # W (new laser power for transfer learning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def thermal_conductivity(T):
    """Normalized thermal conductivity as a function of temperature."""
    return torch.where(T <= T_liquidus, 1.0 + 0.5 * (T - T_inf) / (T_liquidus - T_inf), 0.8)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01),
            nn.Linear(100, 1)
        )

    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.net(inputs)

def input_transform(x, y, z, t):
    """Normalize inputs to [-1, 1], preserving gradient tracking."""
    x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    z = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
    t = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    return x, y, z, t

def output_transform(u_scaled):
    """Transform network output to physical temperature range."""
    return T_inf + T_range * torch.sigmoid(u_scaled)

def heat_source(x, y, z, t, P, epoch=None):
    """Gaussian heat source with proper scaling."""
    x_phys = x
    y_phys = y
    z_phys = z
    t_phys = t
    
    x_laser = torch.where(t_phys <= 2.0, v * t_phys, torch.zeros_like(t_phys))
    r2_x = (x_phys - x_laser) ** 2 / (r_x ** 2 * 2)
    r2_y = (y_phys - 0.01) ** 2 / (r_y ** 2 * 2)
    r2_z = (z_phys - L_z) ** 2 / (r_z ** 2 * 2)
    
    Q = eta * P * torch.exp(-(r2_x + r2_y + r2_z)) * (t_phys <= 2.0).float() * 1e6  # W/m³
    Q = torch.clamp(Q, 0, 1e7)
    if epoch == 0:
        print(f"Heat Source Debug: t_phys min/max: {t_phys.min():.2f}/{t_phys.max():.2f}, "
              f"x_phys min/max: {x_phys.min():.2f}/{x_phys.max():.2f}, Q max: {Q.max():.2e}")
    return Q

def pde_residual(model, x, y, z, t, P, epoch=None):
    x_model, y_model, z_model, t_model = input_transform(x, y, z, t)
    x_model, y_model, z_model, t_model = x_model.to(device), y_model.to(device), z_model.to(device), t_model.to(device)
    u_scaled = model(x_model, y_model, z_model, t_model)
    u = output_transform(u_scaled)
    u_t = torch.autograd.grad(u, t_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / t_total)
    u_x = torch.autograd.grad(u, x_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_x)
    u_xx = torch.autograd.grad(u_x, x_model, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] * (2 / L_x)
    u_y = torch.autograd.grad(u, y_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_y)
    u_yy = torch.autograd.grad(u_y, y_model, grad_outputs=torch.ones_like(u_y), create_graph=True)[0] * (2 / L_y)
    u_z = torch.autograd.grad(u, z_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_z)
    u_zz = torch.autograd.grad(u_z, z_model, grad_outputs=torch.ones_like(u_z), create_graph=True)[0] * (2 / L_z)

    k = thermal_conductivity(u) * k_max
    Q = heat_source(x, y, z, t, P, epoch)
    
    latent_heat = 225000.0  # J/kg
    phase_factor = torch.where((u >= T_liquidus) & (u <= T_solidus), latent_heat * rho / t_total, 0.0)  # W/m³
    u_t_adjusted = u_t - phase_factor / (rho * cp)  # K/s

    f = rho * cp * u_t_adjusted - k * (u_xx + u_yy + u_zz) - Q
    f_scale = rho * cp * T_range / t_total  # ~1.54e7 W/m³
    f_normalized = f / f_scale
    
    if epoch == 0:
        print(f"PDE Debug: u_t min/max: {u_t.min():.2e}/{u_t.max():.2e}, "
              f"u_xx min/max: {u_xx.min():.2e}/{u_xx.max():.2e}, f min/max: {f.min():.2e}/{f.max():.2e}")
        print(f"u min/max: {u.min().item():.2f}/{u.max().item():.2f}")
        print(f"Phase factor max: {phase_factor.max():.2e}")
        print(f"rho*cp*u_t min/max: {(rho * cp * u_t_adjusted).min().item():.2e}/{(rho * cp * u_t_adjusted).max().item():.2e}")
        print(f"k*(u_xx+u_yy+u_zz) min/max: {(k * (u_xx + u_yy + u_zz)).min().item():.2e}/{(k * (u_xx + u_yy + u_zz)).max().item():.2e}")
        print(f"Q min/max: {Q.min():.2e}/{Q.max():.2e}")
        print(f"f_normalized min/max: {f_normalized.min():.2e}/{f_normalized.max():.2e}")
    if torch.any(torch.isnan(f_normalized)) or torch.any(torch.isinf(f_normalized)):
        print(f"NaN or Inf in PDE residual at epoch {epoch}")
    return f_normalized

def bc_residual(model, x, y, z, t, P, epoch=None):
    x_model, y_model, z_model, t_model = input_transform(x, y, z, t)
    x_model, y_model, z_model, t_model = x_model.to(device), y_model.to(device), z_model.to(device), t_model.to(device)
    u_scaled = model(x_model, y_model, z_model, t_model)
    u = output_transform(u_scaled)
    u = torch.clamp(u, T_inf, T_inf + T_range)
    u_x = torch.autograd.grad(u, x_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_x)
    u_y = torch.autograd.grad(u, y_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_y)
    u_z = torch.autograd.grad(u, z_model, grad_outputs=torch.ones_like(u), create_graph=True)[0] * (2 / L_z)

    k = thermal_conductivity(u) * k_max
    Q = heat_source(x, y, z, t, P, epoch)
    if epoch == 0:
        print(f"BC Debug: Q max: {Q.max().item():.2e}, u max: {u.max().item():.2f}")

    conv_term = h * (u - T_inf)
    rad_term = epsilon * sigma * (u**4 - T_inf**4)
    Q_scaled = Q * (L_z / 2) / (T_range * h)
    bc_x_neg = (-k * u_x - conv_term - rad_term) * (x_model <= -0.999).float()
    bc_x_pos = (k * u_x - conv_term - rad_term) * (x_model >= 0.999).float()
    bc_y_neg = (-k * u_y - conv_term - rad_term) * (y_model <= -0.999).float()
    bc_y_pos = (k * u_y - conv_term - rad_term) * (y_model >= 0.999).float()
    bc_z_neg = (-k * u_z - conv_term - rad_term) * (z_model <= -0.999).float()
    bc_z_pos = (-k * u_z - conv_term - rad_term + Q_scaled) * (z_model >= 0.999).float()

    f_bc = bc_x_neg + bc_x_pos + bc_y_neg + bc_y_pos + bc_z_neg + bc_z_pos
    f_bc_normalized = f_bc / (h * T_range)  # Reverted to ~0.01–0.1 scale
    if torch.any(torch.isnan(f_bc_normalized)) or torch.any(torch.isinf(f_bc_normalized)):
        print(f"NaN or Inf in BC residual at epoch {epoch}")
    return f_bc_normalized

def ic_residual(model, x, y, z, t, epoch=None):
    x_model, y_model, z_model, t_model = input_transform(x, y, z, t)
    x_model, y_model, z_model, t_model = x_model.to(device), y_model.to(device), z_model.to(device), t_model.to(device)
    u_scaled = model(x_model, y_model, z_model, t_model)
    u = output_transform(u_scaled)
    f_ic = u - T_inf
    f_ic_normalized = f_ic / T_range
    if torch.any(torch.isnan(f_ic_normalized)) or torch.any(torch.isinf(f_ic_normalized)):
        print(f"NaN or Inf in IC residual at epoch {epoch}")
    return f_ic_normalized
