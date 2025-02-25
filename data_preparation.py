import torch
from pinn_model import input_transform  # Import the scaling function from your PINN model

# ---------------------------- Data Preparation Functions ----------------------------

def generate_uniform_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N, dense_region=None, dense_fraction=0.7):
    """
    Generate uniform random points, with denser sampling near a region (e.g., laser path).
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max: Domain bounds (meters, seconds)
        N: Number of points to generate
        dense_region: Tuple (x_range, y_range, z_range, t_range) for denser sampling
        dense_fraction: Fraction of points in dense region (default 0.7)
    
    Returns:
        x, y, z, t: Tensors of shape (N, 1)
    """
    N_dense, N_rand = int(N * dense_fraction), N - int(N * dense_fraction)
    
    if dense_region:
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

def generate_boundary_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N_per_face=500, N_top=1500):
    total_points = N_per_face * 5 + N_top
    assert total_points == 4000
    
    # x = 0 m
    x_b_xneg = torch.ones(N_per_face, 1) * x_min
    y_b_xneg = torch.FloatTensor(N_per_face, 1).uniform_(y_min, y_max)
    z_b_xneg = torch.FloatTensor(N_per_face, 1).uniform_(z_min, z_max)
    t_b_xneg = torch.FloatTensor(N_per_face, 1).uniform_(t_min, t_max)
    
    # x = 0.04 m
    x_b_xpos = torch.ones(N_per_face, 1) * x_max
    y_b_xpos = torch.FloatTensor(N_per_face, 1).uniform_(y_min, y_max)
    z_b_xpos = torch.FloatTensor(N_per_face, 1).uniform_(z_min, z_max)
    t_b_xpos = torch.FloatTensor(N_per_face, 1).uniform_(t_min, t_max)
    
    # y = 0 m
    x_b_yneg = torch.FloatTensor(N_per_face, 1).uniform_(x_min, x_max)
    y_b_yneg = torch.ones(N_per_face, 1) * y_min
    z_b_yneg = torch.FloatTensor(N_per_face, 1).uniform_(z_min, z_max)
    t_b_yneg = torch.FloatTensor(N_per_face, 1).uniform_(t_min, t_max)
    
    # y = 0.02 m
    x_b_ypos = torch.FloatTensor(N_per_face, 1).uniform_(x_min, x_max)
    y_b_ypos = torch.ones(N_per_face, 1) * y_max
    z_b_ypos = torch.FloatTensor(N_per_face, 1).uniform_(z_min, z_max)
    t_b_ypos = torch.FloatTensor(N_per_face, 1).uniform_(t_min, t_max)
    
    # z = 0 m
    x_b_zneg = torch.FloatTensor(N_per_face, 1).uniform_(x_min, x_max)
    y_b_zneg = torch.FloatTensor(N_per_face, 1).uniform_(y_min, y_max)
    z_b_zneg = torch.ones(N_per_face, 1) * z_min
    t_b_zneg = torch.FloatTensor(N_per_face, 1).uniform_(t_min, t_max)
    
    # z = 0.005 m (top surface), split into deposition and cooling
    N_top_deposition = N_top // 2  # 750 points during deposition
    N_top_cooling = N_top - N_top_deposition  # 750 points during cooling
    x_b_zpos_dep = torch.FloatTensor(N_top_deposition, 1).uniform_(0.0, 0.04)  # Full x-range
    y_b_zpos_dep = torch.FloatTensor(N_top_deposition, 1).uniform_(0.009, 0.011)
    z_b_zpos_dep = torch.ones(N_top_deposition, 1) * z_max
    t_b_zpos_dep = torch.FloatTensor(N_top_deposition, 1).uniform_(0.0, 2.0)  # Deposition only
    
    x_b_zpos_cool = torch.FloatTensor(N_top_cooling, 1).uniform_(0.0, 0.04)
    y_b_zpos_cool = torch.FloatTensor(N_top_cooling, 1).uniform_(0.009, 0.011)
    z_b_zpos_cool = torch.ones(N_top_cooling, 1) * z_max
    t_b_zpos_cool = torch.FloatTensor(N_top_cooling, 1).uniform_(2.0, t_max)  # Cooling
    
    x_b_zpos = torch.cat([x_b_zpos_dep, x_b_zpos_cool])
    y_b_zpos = torch.cat([y_b_zpos_dep, y_b_zpos_cool])
    z_b_zpos = torch.cat([z_b_zpos_dep, z_b_zpos_cool])
    t_b_zpos = torch.cat([t_b_zpos_dep, t_b_zpos_cool])
    
    x_b = torch.cat([x_b_xneg, x_b_xpos, x_b_yneg, x_b_ypos, x_b_zneg, x_b_zpos])
    y_b = torch.cat([y_b_xneg, y_b_xpos, y_b_yneg, y_b_ypos, y_b_zneg, y_b_zpos])
    z_b = torch.cat([z_b_xneg, x_b_xpos, z_b_yneg, z_b_ypos, z_b_zneg, z_b_zpos])
    t_b = torch.cat([t_b_xneg, t_b_xpos, t_b_yneg, t_b_ypos, t_b_zneg, t_b_zpos])
    
    return x_b, y_b, z_b, t_b

def generate_initial_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, N, dense_fraction=0.5):
    """
    Generate initial condition points at t = 0, with denser sampling near the laser path.
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max, t_min: Domain bounds (meters, seconds)
        N: Number of points to generate
        dense_fraction: Fraction of points near laser path (default 0.5)
    
    Returns:
        x_i, y_i, z_i, t_i: Tensors of shape (N, 1)
    """
    N_dense, N_rand = int(N * dense_fraction), N - int(N * dense_fraction)
    
    # Random points
    x_i_rand = torch.FloatTensor(N_rand, 1).uniform_(x_min, x_max)
    y_i_rand = torch.FloatTensor(N_rand, 1).uniform_(y_min, y_max)
    z_i_rand = torch.FloatTensor(N_rand, 1).uniform_(z_min, z_max)
    t_i_rand = torch.zeros(N_rand, 1)
    
    # Dense points near laser path (x0 = 0.02 m, y0 = 0.01 m)
    x_i_dense = torch.linspace(max(x_min, 0.018), min(x_max, 0.022), N_dense).reshape(-1, 1)
    y_i_dense = torch.linspace(max(y_min, 0.009), min(y_max, 0.011), N_dense).reshape(-1, 1)
    z_i_dense = torch.FloatTensor(N_dense, 1).uniform_(z_min, z_max)
    t_i_dense = torch.zeros(N_dense, 1)
    
    x_i = torch.cat([x_i_dense, x_i_rand])
    y_i = torch.cat([y_i_dense, y_i_rand])
    z_i = torch.cat([z_i_dense, z_i_rand])
    t_i = torch.cat([t_i_dense, t_i_rand])
    
    return x_i, y_i, z_i, t_i

# ---------------------------- Main Data Generation ----------------------------

# Number of points
N_f = 10000  # PDE collocation points
N_b = 4000   # Total boundary points
N_i = 2000   # Initial condition points

# Domain bounds (meters, seconds)
x_min, x_max = 0.0, 0.04
y_min, y_max = 0.0, 0.02
z_min, z_max = 0.0, 0.005
t_min, t_max = 0.0, 102.0

# Generate points
dense_region = (
    (0.018, 0.022),  # x near 0.02 m
    (0.009, 0.011),  # y near 0.01 m
    (0.003, 0.005),  # z widened to capture heat penetration
    (0.0, 2.0)       # t during deposition
)
x_f, y_f, z_f, t_f = generate_uniform_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N_f, dense_region, dense_fraction=0.7)
x_b, y_b, z_b, t_b = generate_boundary_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, N_per_face=500, N_top=1500)
x_i, y_i, z_i, t_i = generate_initial_points(x_min, x_max, y_min, y_max, z_min, z_max, t_min, N_i, dense_fraction=0.5)

# Move data to device with requires_grad=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
x_f = x_f.to(device).requires_grad_(True)
y_f = y_f.to(device).requires_grad_(True)
z_f = z_f.to(device).requires_grad_(True)
t_f = t_f.to(device).requires_grad_(True)
x_b = x_b.to(device).requires_grad_(True)
y_b = y_b.to(device).requires_grad_(True)
z_b = z_b.to(device).requires_grad_(True)
t_b = t_b.to(device).requires_grad_(True)
x_i = x_i.to(device).requires_grad_(True)
y_i = y_i.to(device).requires_grad_(True)
z_i = z_i.to(device).requires_grad_(True)
t_i = t_i.to(device).requires_grad_(True)

# Debug: Check shapes and ranges (in physical units)
print(f"x_f shape: {x_f.shape}, min/max: {x_f.min():.4f}/{x_f.max():.4f}")
print(f"x_b shape: {x_b.shape}, min/max: {x_b.min():.4f}/{x_b.max():.4f}")
print(f"x_i shape: {x_i.shape}, min/max: {x_i.min():.4f}/{x_i.max():.4f}")
print(f"t_i unique values: {t_i.unique()}")  # Should be [0] before normalization

# Save points in physical units
torch.save({
    'x_f': x_f, 'y_f': y_f, 'z_f': z_f, 't_f': t_f,
    'x_b': x_b, 'y_b': y_b, 'z_b': z_b, 't_b': t_b,
    'x_i': x_i, 'y_i': y_i, 'z_i': z_i, 't_i': t_i
}, 'data_points.pt')

# Export variables for use in train_pinn.py
if __name__ == "__main__":
    globals().update({
        'x_f': x_f, 'y_f': y_f, 'z_f': z_f, 't_f': t_f,
        'x_b': x_b, 'y_b': y_b, 'z_b': z_b, 't_b': t_b,
        'x_i': x_i, 'y_i': y_i, 'z_i': z_i, 't_i': t_i,
        'N_f': N_f, 'N_b': N_b, 'N_i': N_i
    })
