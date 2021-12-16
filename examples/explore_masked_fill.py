import torch
from cleanformer.tensors import subsequent_mask

def main():
    #
    L = 5
    sims = torch.rand(size=(L, L))   # (L, L)
    mask = subsequent_mask(L)
    print(sims)
    print(mask)
    # masked_fill
    print(mask == 0)
    out = torch.masked_fill(sims, mask == 0, value=float("-inf"))
    print(out)

if __name__ == '__main__':
    main()