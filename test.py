import torch
from torch.nn import TransformerEncoderLayer
import torch.cuda.profiler as profiler
from torch.cuda import memory_allocated, memory_reserved


def initialize_transformer():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parameters
    embedding_dim = 256
    num_heads = 8
    dff = 512
    activation = 'relu'

    transformer_block = TransformerEncoderLayer(
        d_model=embedding_dim, nhead=num_heads, dim_feedforward=dff, activation=activation, batch_first=True, norm_first=False
    )

    transformer_block = transformer_block.to(device)
    transformer_block.eval()
    
    return transformer_block, device

def get_model_parameters(model):
    for name, tensor in model.state_dict().items():
      print(name, " : ", tensor.numel(), "parameters")

def run_profiler_experiment(model, device, batch_size, num_tokens, embedding_dim):
    data = torch.rand(batch_size, num_tokens, embedding_dim).to(device)
    profiler.start()
    out = model(data)
    profiler.stop()

def memory_utilization(device):
  if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if __name__ == "__main__":
    
    transformer, device = initialize_transformer()
    print(transformer)

    print("#############################Running Profiler##############################")
    
    total_params = get_model_parameters(transformer)
    memory_utilization(device)
    
    batch_size = 1
    num_tokens = 10
    embedding_dim = 256
    
    run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
