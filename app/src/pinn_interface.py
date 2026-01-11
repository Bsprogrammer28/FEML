import torch
import torchphysics as tp
from torchphysics.problem.spaces import Points

class Beam3DPINN():
    def __init__(self, model_path, device='cpu'):
        
        self.device = device

        X = tp.spaces.R1('x')
        Y = tp.spaces.R1('y')
        Z = tp.spaces.R1('z')

        U = tp.spaces.R1('u')  # deflection
        V = tp.spaces.R1('v')  # deflection
        W_ = tp.spaces.R1('w')  # deflection    

        self.model = tp.models.FCN(
            input_space=X*Y*Z,
            output_space=U*V*W_,
            hidden=(128, 128, 128),
            activations=torch.nn.Tanh()
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, points):

        pts = torch.tensor(points, dtype=torch.float32, device=self.device) # Nx3

        tp_points = Points(
            pts, 
            space=self.model.input_space
        )

        out = self.model(tp_points)

        return out.as_tensor.cpu().numpy()