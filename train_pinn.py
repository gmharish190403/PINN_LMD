import torch
import torch.nn as nn
import torch.optim as optim
from pinn_model import PINN, P_base, P_new, pde_residual, bc_residual, ic_residual, input_transform, output_transform, rho, cp, T_range, t_total
from data_preparation import x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, N_f, N_b, N_i
import os
import matplotlib.pyplot as plt

# ---------------------------- Device Setup ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
else:
    print("No GPU available, using CPU")

# ---------------------------- Training Setup ----------------------------
model = PINN().to(device)
optimizer_adam = optim.Adam(model.parameters(), lr=0.005)
epochs_adam = 5000
lr_schedule = {1250, 2500, 3750}
save_path = "pinn_new.pth"

# Loss weights (adjusted for final push)
w_pde, w_bc, w_ic = 10.0, 5000.0, 100.0  # Max BC emphasis

# Lists to store loss values for plotting
pde_losses = []
bc_losses = []
ic_losses = []
total_losses = []

# Force restart from scratch with moderate Xavier initialization
print("🚀 Starting training from scratch with moderate initialization")
for m in model.net.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)

# Debug: Check input shapes
print(f"x_f shape: {x_f.shape}, y_f shape: {y_f.shape}, z_f shape: {z_f.shape}, t_f shape: {t_f.shape}")
print(f"x_b shape: {x_b.shape}, y_b shape: {y_b.shape}, z_b shape: {z_b.shape}, t_b shape: {t_b.shape}")
print(f"x_i shape: {x_i.shape}, y_i shape: {y_i.shape}, z_i shape: {z_i.shape}, t_i shape: {t_i.shape}")

# Validation function
def validate(model, test_in, test_out):
    x, y, z, t = input_transform(test_in[:, 0:1], test_in[:, 1:2], test_in[:, 2:3], test_in[:, 3:4])
    u_pred_scaled = model(x, y, z, t)
    u_pred = output_transform(u_pred_scaled)
    error = torch.mean((u_pred - test_out) ** 2) ** 0.5
    return error.item(), u_pred.item()

# Training function with Adam only
def train_pinn(model, optimizer_adam, epochs_adam, P, save_path, start_epoch, 
               x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, lr_schedule):
    test_in = torch.tensor([[0.02, 0.01, 0.0025, 2.0]], device=device)
    test_out = torch.tensor([[2766.0]], device=device)
    
    batch_size = 1000
    print_interval = 100
    
    print(f"Starting Adam training with P = {P} W...")
    for epoch in range(start_epoch, epochs_adam):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_used = torch.cuda.memory_allocated() / 1024**2
            if mem_used > 3500:
                print(f"Warning: High GPU memory usage ({mem_used:.2f} MB) at epoch {epoch}")

        model.train()
        optimizer_adam.zero_grad()

        # PDE loss
        loss_pde = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(0, N_f, batch_size):
            batch_x_f = x_f[i:i+batch_size]
            batch_y_f = y_f[i:i+batch_size]
            batch_z_f = z_f[i:i+batch_size]
            batch_t_f = t_f[i:i+batch_size]
            f_pde = pde_residual(model, batch_x_f, batch_y_f, batch_z_f, batch_t_f, P, epoch)
            batch_loss = torch.mean(f_pde ** 2)
            loss_pde = loss_pde + batch_loss * (batch_size / N_f)

        # BC loss
        loss_bc = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(0, N_b, batch_size):
            batch_x_b = x_b[i:i+batch_size]
            batch_y_b = y_b[i:i+batch_size]
            batch_z_b = z_b[i:i+batch_size]
            batch_t_b = t_b[i:i+batch_size]
            f_bc = bc_residual(model, batch_x_b, batch_y_b, batch_z_b, batch_t_b, P, epoch)
            batch_loss = torch.mean(f_bc ** 2)
            loss_bc = loss_bc + batch_loss * (batch_size / N_b)

        # IC loss
        loss_ic = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(0, N_i, batch_size):
            batch_x_i = x_i[i:i+batch_size]
            batch_y_i = y_i[i:i+batch_size]
            batch_z_i = z_i[i:i+batch_size]
            batch_t_i = t_i[i:i+batch_size]
            f_ic = ic_residual(model, batch_x_i, batch_y_i, batch_z_i, batch_t_i, epoch)
            batch_loss = torch.mean(f_ic ** 2)
            loss_ic = loss_ic + batch_loss * (batch_size / N_i)

        # Total loss
        loss = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf in loss at epoch {epoch}")
            break

        # Backpropagation and optimization
        loss.backward()
        if epoch % print_interval == 0 or epoch == start_epoch:
            grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))
            error, u_pred = validate(model, test_in, test_out)
            u_all_scaled = model(x_f, y_f, z_f, t_f)
            u_all = output_transform(u_all_scaled)
            u_min, u_max = u_all.min().item(), u_all.max().item()
            print(f"Gradient Norm: {grad_norm.item():.6f}")
            print(f"Raw PDE Loss: {loss_pde.item():.6f}, Effective PDE Loss: {w_pde * loss_pde.item():.6f}")
            print(f"Epoch {epoch}/{epochs_adam}, Loss: {loss.item():.6f}, "
                  f"PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, IC: {loss_ic.item():.6f}, "
                  f"Validation Error: {error:.6f}, Predicted Temp: {u_pred:.2f} K, "
                  f"U Min/Max: {u_min:.2f}/{u_max:.2f} K")
            torch.save(model.state_dict(), f"pinn_epoch{epoch}.pth")
            pde_losses.append(loss_pde.item())
            bc_losses.append(loss_bc.item())
            ic_losses.append(loss_ic.item())
            total_losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer_adam.step()

        if epoch in lr_schedule:
            for param_group in optimizer_adam.param_groups:
                param_group["lr"] *= 0.5
                print(f"Learning rate decayed to {param_group['lr']} at epoch {epoch}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(total_losses)), total_losses, label='Total Loss', color='blue')
    plt.semilogy(range(len(pde_losses)), pde_losses, label='PDE Loss', color='green')
    plt.semilogy(range(len(bc_losses)), bc_losses, label='BC Loss', color='red')
    plt.semilogy(range(len(ic_losses)), ic_losses, label='IC Loss', color='orange')
    plt.xlabel('Epochs (Adam)')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Loss Curve (P = {P} W)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_P_{P}.png')
    plt.show()

# ---------------------------- Main Training Loop ----------------------------
if __name__ == "__main__":
    print("Starting training with P = 2400 W from scratch...")
    train_pinn(model, optimizer_adam, epochs_adam, P_new, save_path, 0,
               x_f, y_f, z_f, t_f, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, lr_schedule)

    print("✅ Training completed!")
