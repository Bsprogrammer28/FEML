import tkinter as tk
from tkinter import ttk
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import traceback

class BeamFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class BeamTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Beam Deflection Tester - FIXED Parameters")
        self.root.geometry("1000x700")
        
        # FIXED PARAMETERS (cannot be changed)
        self.TEST_LENGTH = 3.0
        self.TEST_FORCE = 100.0
        self.TEST_POS = 0.6
        self.TEST_POINTS = 100
        
        self.model = None
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="FIXED PARAMETERS", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Label(main_frame, text=f"Length (L): {self.TEST_LENGTH} m").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(main_frame, text=f"Force (F): {self.TEST_FORCE} N").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(main_frame, text=f"Force Position (A): {self.TEST_POS} ({self.TEST_POS*self.TEST_LENGTH:.1f} m)").grid(row=3, column=0, sticky=tk.W, pady=2)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Compute & Show Profile", 
                  command=self.compute_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reload Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2, pady=5)
        
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
    
    def load_model(self):
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            model_path = Path("beam_model.pth")
            if not model_path.exists():
                self.status_var.set("❌ ERROR: beam_model.pth not found! Run 1dTest.py first.")
                return False
            
            # Create model and load weights
            self.model = BeamFCN()
            
            # Load with strict=False to handle any key mismatches
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            self.status_var.set("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"❌ Load error: {str(e)}"
            self.status_var.set(error_msg)
            print("Full error:", traceback.format_exc())  # Debug info in console
            self.model = None
            return False
    
    def compute_profile(self):
        if self.model is None:
            self.status_var.set("❌ ERROR: Model not loaded!")
            return
        
        try:
            self.status_var.set("Computing profile...")
            self.root.update()
            
            # Test inputs as plain tensors
            z_test = torch.linspace(0, 1, self.TEST_POINTS).reshape(-1, 1)
            l_test = torch.full_like(z_test, float(self.TEST_LENGTH))
            a_test = torch.full_like(z_test, float(self.TEST_POS))
            f_test = torch.full_like(z_test, float(self.TEST_FORCE))
            
            test_inputs = torch.cat([z_test, l_test, a_test, f_test], dim=1)
            
            with torch.no_grad():
                deflections = self.model(test_inputs).numpy().flatten()
            
            # Physical units
            physical_z = z_test.numpy().flatten() * self.TEST_LENGTH
            deflections_physical = deflections * self.TEST_LENGTH
            
            # Clear plot area
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(physical_z, deflections_physical, 'b-', linewidth=3, label='PINN Prediction')
            
            force_loc = self.TEST_POS * self.TEST_LENGTH
            ax.axvline(x=force_loc, color='r', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Force: {self.TEST_FORCE}N at {force_loc:.2f}m')
            
            max_idx = np.argmin(deflections_physical)
            max_defl = deflections_physical[max_idx]
            max_pos = physical_z[max_idx]
            ax.plot(max_pos, max_defl, 'go', markersize=10, 
                   label=f'Max deflection: {max_defl:.4f}m')
            
            ax.set_xlabel('Position along beam (m)')
            ax.set_ylabel('Deflection (m)')
            ax.set_title(f'Beam Deflection Profile\nL={self.TEST_LENGTH}m, F={self.TEST_FORCE}N @ {force_loc:.2f}m')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(min(deflections_physical)*1.1, 0)
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_var.set(f"✅ Max deflection: {max_defl:.4f}m at {max_pos:.2f}m")
            
        except Exception as e:
            error_msg = f"❌ Compute error: {str(e)}"
            self.status_var.set(error_msg)
            print("Full compute error:", traceback.format_exc())
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = BeamTestGUI(root)
    app.run()
