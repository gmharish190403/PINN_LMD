import torch
from pinn_model import PINN, input_transform, output_transform, device

# ---------------------------- Model Evaluation ----------------------------
def evaluate_model(model, x, y, z, t):
    """Evaluates the trained PINN model at given (x, y, z, t) points."""
    model.eval()  # Set model to evaluation mode

    # Transform inputs to match training normalization
    x, y, z, t = input_transform(x, y, z, t)
    x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)

    with torch.no_grad():  # No need for gradient computation
        temp_scaled = model(x, y, z, t)  # Model prediction (scaled temperature)
        temperature = output_transform(temp_scaled)  # Convert back to real temperature (Kelvin)

    return temperature

# ---------------------------- Testing Example ----------------------------
if __name__ == "__main__":
    # Load trained model
    model = PINN().to(device)
    model.load_state_dict(torch.load("pinn_model.pth", map_location=device, weights_only=True))
    print("✅ Model loaded successfully!")

    # Define test point (center of substrate at t=2s, in meters)
    x_test = torch.tensor([[0.02]], device=device)  # Center of x (20 mm = 0.02 m)
    y_test = torch.tensor([[0.01]], device=device)  # Center of y (10 mm = 0.01 m)
    z_test = torch.tensor([[0.0025]], device=device)  # Center of z (2.5 mm = 0.0025 m)
    t_test = torch.tensor([[2.0]], device=device)  # Time = 2s (end of deposition)

    # Get prediction
    temp_pred = evaluate_model(model, x_test, y_test, z_test, t_test)

    # Print result
    print(f"🔥 Predicted Temperature at (x=20mm, y=10mm, z=2.5mm, t=2s): {temp_pred.item():.2f} K")

    # Optional: Add validation against expected FEM data from Table 2
    expected_temp = 1690.0  # Example liquidus temp, adjust from paper/FEM
    error = abs(temp_pred.item() - expected_temp)
    print(f"🔍 Error compared to expected (e.g., 1690 K): {error:.2f} K")
