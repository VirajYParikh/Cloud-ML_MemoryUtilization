import torch
from torch.nn import TransformerEncoderLayer
import torch.cuda.profiler as profiler
from torch.cuda import memory_allocated, memory_reserved
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F


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
    print("#############################Profiler Results##############################")
    # print(profiler.key_averages().table(sort_by="cuda_time_total"))
    output = F.log_softmax(out, dim=1)
    return output
    # memory_usage = memory_utilization(device)
    
    
    
    

def memory_utilization(device):
  if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return round(torch.cuda.memory_allocated(0)/1024**3,1) + round(torch.cuda.memory_reserved(0)/1024**3,1)

if __name__ == "__main__":

    num_tokens = 8
    embedding_dim = 256
    min_batch_size = 128
    max_batch_size = 4096
    num_heads = 8

    transformer, device = initialize_transformer()
    
    print(transformer)
    print("#############################Running Profiler##############################")

    batch_sizes = list(range(1, max_batch_size + 1))
    memory_usage = []

    gpu_results = []
    for batch_size in range(min_batch_size, max_batch_size + 1, 64):

        output = run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
        print("Output: " + output)
        # gpu_results.append(gpu_result)


    # print(gpu_results)

    print("#############################Model Parameters##############################")
    
    total_params = get_model_parameters(transformer)
    
    # batch_size = 1
    # num_tokens = 10
    # embedding_dim = 256
    
    # run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
