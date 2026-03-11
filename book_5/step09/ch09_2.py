import torch



def _pos_encoding(t, output_dim, device='cpu'):
    D = output_dim
    v = torch.zeros(D, device=device)
    
    i = torch.arange(0, D, device=device)
    div_term = 10000 ** (i / D)
    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(ts, output_dim, device='cpu'):
    batch_size = ts.shape[0]
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v

if __name__ == "__main__":
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    output_dim = 16
    pos_encoding1 = pos_encoding(t, output_dim)
    print(pos_encoding1.shape)