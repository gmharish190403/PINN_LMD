import torch
from pinn_model import input_transform  # Import the scaling function from your PINN model

# ---------------------------- Data Preparation Functions ----------------------------

def generate_uniform_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N, dense_region=None):
    """
    Generate uniform random points, with optional denser sampling near a region (e.g., laser path).
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max: Domain bounds (in meters, seconds)
        N: Number of points to generate
        dense_region: Optional tuple (x_range, y_range, z_range, t_range) for denser sampling
    
    Returns:
        x, y, z, t: Tensors of shape (N, 1)
    """
    if dense_region:
        N_dense, N_rand = N // 2, N - N // 2
        x_dense = torch.linspace(max(x_min, dense_region[0][0]), min(x_max, dense_region[0][1]), N_dense).reshape(-1, 1)
        y_dense = torch.linspace(max(y_min, dense_region[1][0]), min(y_max, dense_region[1][1]), N_dense).reshape(-1, 1)
        z_dense = torch.FloatTensor(N_dense, 1).uniform_(max(z_min, dense_region[2][0]), min(z_max, dense_region[2][1]))
        t_dense = torch.FloatTensor(N_dense, 1).uniform_(max(t_min, dense_region[3][0]), min(t_max, dense_region[3][1]))
        
        x_rand = torch.FloatTensor(N_rand, 1).uniform_(x_min, x_max)
        y_rand = torch.FloatTensor(N_rand, 1).uniform_(y_min, y_max)
        z_rand = torch.FloatTensor(N_rand, 1).uniform_(z_min, z_max)
        t_rand = torch.FloatTensor(N_rand, 1).uniform_(t_min, t_max)
        
        x = torch.cat([x_dense, x_rand])
        y = torch.cat([y_dense, y_rand])
        z = torch.cat([z_dense, z_rand])
        t = torch.cat([t_dense, t_rand])
    else:
        x = torch.FloatTensor(N, 1).uniform_(x_min, x_max)
        y = torch.FloatTensor(N, 1).uniform_(y_min, y_max)
        z = torch.FloatTensor(N, 1).uniform_(z_min, z_max)
        t = torch.FloatTensor(N, 1).uniform_(t_min, t_max)
    
    return x, y, z, t

def generate_boundary_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N_per_face):
    """
    Generate boundary points for each face, with denser sampling on the top surface near the laser path.
    Ensures exactly 4000 total points.
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max: Domain bounds (in meters, seconds)
        N_per_face: Base number of points per face (adjusted to ensure 4000 total)
    
    Returns:
        x_b, y_b, z_b, t_b: Tensors of shape (4000, 1)
    """
    # Calculate points per face to ensure exactly 4000 total
    base_points = 666  # 666 * 5 = 3330, adjust 6th face for 4000 total
    extra_points = 4000 - (base_points * 5)  # 670 for the 6th face (top surface)
    
    # x = 0 m (-1 in normalized)
    x_b_xneg = torch.ones(base_points, 1) * x_min
    y_b_xneg = torch.FloatTensor(base_points, 1).uniform_(y_min, y_max)
    z_b_xneg = torch.FloatTensor(base_points, 1).uniform_(z_min, z_max)
    t_b_xneg = torch.FloatTensor(base_points, 1).uniform_(t_min, t_max)
    
    # x = 0.04 m (1 in normalized)
    x_b_xpos = torch.ones(base_points, 1) * x_max
    y_b_xpos = torch.FloatTensor(base_points, 1).uniform_(y_min, y_max)
    z_b_xpos = torch.FloatTensor(base_points, 1).uniform_(z_min, z_max)
    t_b_xpos = torch.FloatTensor(base_points, 1).uniform_(t_min, t_max)
    
    # y = 0 m (-1 in normalized)
    x_b_yneg = torch.FloatTensor(base_points, 1).uniform_(x_min, x_max)
    y_b_yneg = torch.ones(base_points, 1) * y_min
    z_b_yneg = torch.FloatTensor(base_points, 1).uniform_(z_min, z_max)
    t_b_yneg = torch.FloatTensor(base_points, 1).uniform_(t_min, t_max)
    
    # y = 0.02 m (1 in normalized)
    x_b_ypos = torch.FloatTensor(base_points, 1).uniform_(x_min, x_max)
    y_b_ypos = torch.ones(base_points, 1) * y_max
    z_b_ypos = torch.FloatTensor(base_points, 1).uniform_(z_min, z_max)
    t_b_ypos = torch.FloatTensor(base_points, 1).uniform_(t_min, t_max)
    
    # z = 0 m (-1 in normalized)
    x_b_zneg = torch.FloatTensor(base_points, 1).uniform_(x_min, x_max)
    y_b_zneg = torch.FloatTensor(base_points, 1).uniform_(y_min, y_max)
    z_b_zneg = torch.ones(base_points, 1) * z_min
    t_b_zneg = torch.FloatTensor(base_points, 1).uniform_(t_min, t_max)
    
    # z = 0.005 m (1 in normalized), denser near laser path during deposition, with extra points
    x_b_zpos = torch.FloatTensor(extra_points, 1).uniform_(max(x_min, 0.018), min(x_max, 0.022))  # Near 0.02 m (20 mm)
    y_b_zpos = torch.FloatTensor(extra_points, 1).uniform_(max(y_min, 0.009), min(y_max, 0.011))  # Near 0.01 m (10 mm)
    z_b_zpos = torch.ones(extra_points, 1) * z_max
    t_b_zpos = torch.FloatTensor(extra_points, 1).uniform_(t_min, min(t_max, 2.0))  # Only during deposition (0–2 s)
    
    x_b = torch.cat([x_b_xneg, x_b_xpos, x_b_yneg, x_b_ypos, x_b_zneg, x_b_zpos])[:4000]
    y_b = torch.cat([y_b_xneg, y_b_xpos, y_b_yneg, y_b_ypos, y_b_zneg, y_b_zpos])[:4000]
    z_b = torch.cat([z_b_xneg, z_b_xpos, z_b_yneg, z_b_ypos, z_b_zneg, z_b_zpos])[:4000]
    t_b = torch.cat([t_b_xneg, t_b_xpos, t_b_yneg, t_b_ypos, t_b_zneg, t_b_zpos])[:4000]
    
    return x_b, y_b, z_b, t_b

def generate_initial_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, N):
    """
    Generate initial condition points at t = 0, with denser sampling near the laser path.
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max, t_min: Domain bounds (in meters, seconds)
        N: Number of points to generate
    
    Returns:
        x_i, y_i, z_i, t_i: Tensors of shape (N, 1)
    """
    # Random points
    x_i_rand = torch.FloatTensor(N // 2, 1).uniform_(x_min, x_max)
    y_i_rand = torch.FloatTensor(N // 2, 1).uniform_(y_min, y_max)
    z_i_rand = torch.FloatTensor(N // 2, 1).uniform_(z_min, z_max)
    t_i_rand = torch.zeros(N // 2, 1)
    
    # Dense points near laser path (x0 = 0.02 m, y0 = 0.01 m)
    x_i_dense = torch.linspace(max(x_min, 0.018), min(x_max, 0.022), N // 2).reshape(-1, 1)  # Around 0.02 m (20 mm)
    y_i_dense = torch.linspace(max(y_min, 0.009), min(y_max, 0.011), N // 2).reshape(-1, 1)  # Around 0.01 m (10 mm)
    z_i_dense = torch.FloatTensor(N // 2, 1).uniform_(z_min, z_max)
    t_i_dense = torch.zeros(N // 2, 1)
    
    x_i = torch.cat([x_i_dense, x_i_rand])
    y_i = torch.cat([y_i_dense, y_i_rand])
    z_i = torch.cat([z_i_dense, z_i_rand])
    t_i = torch.cat([t_i_dense, t_i_rand])
    
    return x_i, y_i, z_i, t_i

# ---------------------------- Main Data Generation ----------------------------

# Number of points (aligned with paper and your setup)
N_f = 20000  # PDE collocation points
N_b = 4000   # Total boundary points (exactly 4000)
N_i = 2000   # Initial condition points

# Domain bounds (same as PINN model, in meters)
x_min, x_max = 0.0, 0.04  # meters
y_min, y_max = 0.0, 0.02
z_min, z_max = 0.0, 0.005
t_min, t_max = 0.0, 102.0  # seconds

# Generate points
# PDE points, denser near laser path during deposition
dense_region = (
    (0.018, 0.022),  # x range near 0.02 m (20 mm)
    (0.009, 0.011),  # y range near 0.01 m (10 mm)
    (0.004, 0.005),  # z range near 0.005 m (5 mm)
    (0.0, 2.0)       # t range during deposition (0–2 s)
)
x_f, y_f, z_f, t_f = generate_uniform_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N_f, dense_region=dense_region)

# Boundary points, exactly 4000 total
x_b, y_b, z_b, t_b = generate_boundary_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, 666)  # Base points per face
assert x_b.shape[0] == N_b, f"Expected {N_b} boundary points, got {x_b.shape[0]}"

# Initial points at t = 0, denser near laser path
x_i, y_i, z_i, t_i = generate_initial_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, N_i)

# Apply scaling (normalize to [-1, 1])
x_f, y_f, z_f, t_f = input_transform(x_f, y_f, z_f, t_f)
x_b, y_b, z_b, t_b = input_transform(x_b, y_b, z_b, t_b)
x_i, y_i, z_i, t_i = input_transform(x_i, y_i, z_i, t_i)

# Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
x_f, y_f, z_f, t_f = x_f.to(device), y_f.to(device), z_f.to(device), t_f.to(device)
x_b, y_b, z_b, t_b = x_b.to(device), y_b.to(device), z_b.to(device), t_b.to(device)
x_i, y_i, z_i, t_i = x_i.to(device), y_i.to(device), z_i.to(device), t_i.to(device)

# Debug: Check shapes and ranges
print(f"x_f shape: {x_f.shape}, min/max: {x_f.min():.4f}/{x_f.max():.4f}")
print(f"x_b shape: {x_b.shape}, min/max: {x_b.min():.4f}/{x_b.max():.4f}")
print(f"x_i shape: {x_i.shape}, min/max: {x_i.min():.4f}/{x_i.max():.4f}")

# Export variables for use in train_pinn.py
if __name__ == "__main__":
    # Define global variables for import
    globals().update({
        'x_f': x_f, 'y_f': y_f, 'z_f': z_f, 't_f': t_f,
        'x_b': x_b, 'y_b': y_b, 'z_b': z_b, 't_b': t_b,
        'x_i': x_i, 'y_i': y_i, 'z_i': z_i, 't_i': t_i,
        'N_f': N_f, 'N_b': N_b, 'N_i': N_i
    })
