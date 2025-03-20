import torch
from tictoc import tictoc as tt




if __name__ == '__main__':

    tt.tic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch is using '{device}'")
    tt.toc()
    print(f'{tt:.6f}')

