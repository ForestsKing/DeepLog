import torch

from exp.exp import EXP
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = EXP(g=4, epochs=400, generate=False, batch_size=32, w=10, patience=20, lr=0.001, device=device)
    exp.fit()
    exp.predict(pred_load=True, model_load=True)
