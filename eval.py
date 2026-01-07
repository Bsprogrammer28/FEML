import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import torch
import torchphysics as tp
import numpy as np

# --- 1. Load Model with 4 Params ---
Z = tp.spaces.R1('z')
L = tp.spaces.R1('l')
A = tp.spaces.R1('a')
P = tp.spaces.R1('p') # New parameter
U = tp.spaces.R1('u')

# Input space must match training: Z*L*A*P
model = tp.models.FCN(input_space=Z*L*A*P, output_space=U, hidden=(128, 128, 128, 128))

try:
    model.load_state_dict(torch.load("parametric_beam_force.pth"))
    model.eval()
    print("4-Parameter Model Loaded Successfully.")
except FileNotFoundError:
    print("Error: 'parametric_beam_force.pth' not found. Run the training script first!")
    exit()

# --- 2. GUI Application ---
class InstantBeamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Instant PINN Simulator (Variable Force)")
        self.root.geometry("900x700")

        # Input Frame
        input_frame = ttk.Frame(root, padding="20")
        input_frame.pack(fill=tk.X)

        # Length Input
        ttk.Label(input_frame, text="Length (1-5m):").grid(row=0, column=0, padx=5)
        self.ent_len = ttk.Entry(input_frame, width=10)
        self.ent_len.insert(0, "3.0")
        self.ent_len.grid(row=0, column=1, padx=5)

        # Load Position Input
        ttk.Label(input_frame, text="Load Pos (m):").grid(row=0, column=2, padx=5)
        self.ent_pos = ttk.Entry(input_frame, width=10)
        self.ent_pos.insert(0, "2.5")
        self.ent_pos.grid(row=0, column=3, padx=5)

        # Force Input
        ttk.Label(input_frame, text="Force (-100 to 100):").grid(row=0, column=4, padx=5)
        self.ent_force = ttk.Entry(input_frame, width=10)
        self.ent_force.insert(0, "-50.0")
        self.ent_force.grid(row=0, column=5, padx=5)

        # Button
        self.btn_run = ttk.Button(input_frame, text="Predict", command=self.predict)
        self.btn_run.grid(row=0, column=6, padx=20)

        self.lbl_error = ttk.Label(input_frame, text="", foreground="red")
        self.lbl_error.grid(row=1, column=0, columnspan=7)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def predict(self):
        try:
            # Get Inputs
            len_val = float(self.ent_len.get())
            pos_val = float(self.ent_pos.get())
            force_val = float(self.ent_force.get())

            # Validation
            if not (1.0 <= len_val <= 5.0):
                self.lbl_error.config(text="Error: Length must be 1.0 - 5.0")
                return
            if not (0.0 <= pos_val <= len_val):
                self.lbl_error.config(text="Error: Position must be inside beam")
                return
            self.lbl_error.config(text="")

            # Prepare Tensors
            n_points = 200
            z_np = np.linspace(0, 1, n_points)
            a_rel = pos_val / len_val

            # Create columns for all 4 inputs
            z_tensor = torch.tensor(z_np, dtype=torch.float32).reshape(-1, 1)
            l_tensor = torch.full((n_points, 1), len_val)
            a_tensor = torch.full((n_points, 1), a_rel)
            p_tensor = torch.full((n_points, 1), force_val) # Constant force column

            # Combine inputs
            model_inputs = tp.spaces.Points({
                Z: z_tensor, 
                L: l_tensor, 
                A: a_tensor, 
                P: p_tensor
            }, Z*L*A*P)

            # Predict
            with torch.no_grad():
                u_pred = model(model_inputs).numpy()

            # Plot
            x_real = z_np * len_val
            self.ax.clear()
            self.ax.plot(x_real, u_pred, 'r-', linewidth=3, label='Prediction')
            
            # Dynamic Arrow for Force
            # Arrow points DOWN for negative force, UP for positive
            arrow_dy = -1 if force_val < 0 else 1
            arrow_start = np.max(u_pred) + 0.5 if force_val < 0 else np.min(u_pred) - 0.5
            
            # Simple visual marker
            self.ax.annotate(f"Force: {force_val}", 
                             xy=(pos_val, 0), 
                             xytext=(pos_val, arrow_dy * 2),
                             arrowprops=dict(facecolor='blue', shrink=0.05),
                             ha='center')

            self.ax.set_title(f"Prediction (L={len_val}, Force={force_val})")
            self.ax.set_xlabel("Position (m)")
            self.ax.set_ylabel("Deflection")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()

        except ValueError:
            self.lbl_error.config(text="Invalid numbers")

if __name__ == "__main__":
    root = tk.Tk()
    app = InstantBeamApp(root)
    root.mainloop()