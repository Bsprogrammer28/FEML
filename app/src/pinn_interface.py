import torch
import numpy as np
import torchphysics as tp
from torchphysics.problem.spaces.points import Points

X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
Z = tp.spaces.R1('z')
Fx = tp.spaces.R1('fx')
Fy = tp.spaces.R1('fy')
Fz = tp.spaces.R1('fz')

U = tp.spaces.R1('u')
V = tp.spaces.R1('v')
W = tp.spaces.R1('w')

class Beam3DPINN:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        self.model = tp.models.FCN(
            input_space=X * Y * Z * Fx * Fy * Fz,
            output_space=U * V * W,
            hidden=(128, 128, 128),
            activations=torch.nn.Tanh()
        ).to(self.device)

        self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        points: (N,3) normalized coordinates
        returns: (N,3) displacement [u,v,w]
        """

        if points.shape[1] != 3:
            raise ValueError("Input points must be (N,3)")

        N = points.shape[0]

        # Zero traction for preview
        fx = np.zeros((N, 1))
        fy = np.zeros((N, 1))
        fz = np.zeros((N, 1))

        data = np.hstack([points, fx, fy, fz])
        tensor = torch.tensor(
            data, dtype=torch.float32, device=self.device
        )

        pts = Points(
            tensor,
            space=X * Y * Z * Fx * Fy * Fz
        )

        with torch.no_grad():
            out = self.model(pts)

        return out.as_tensor.detach().cpu().numpy()

