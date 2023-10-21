import torch
from torch.nn import TransformerEncoderLayer
import torch.cuda.profiler as profiler
from torch.cuda import memory_allocated, memory_reserved
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


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
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(data)
    cpu_time = prof.key_averages().self_gpu_time_total
    memory_usage = prof.key_averages().device_memory_usage
    print("GPU Time: ", cpu_time)
    print("Memory Usage: ", memory_usage)
    
    profiler.stop()
    return cpu_time, memory_usage
    
    
    

def memory_utilization(device):
  if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if __name__ == "__main__":

    num_tokens = 8
    embedding_dim = 256
    max_batch_size = 10
    num_heads = 8

    transformer, device = initialize_transformer()
    print("#############################Model Parameters##############################")
    print(transformer)
    print("#############################Running Profiler##############################")

    batch_sizes = list(range(1, max_batch_size + 1))
    memory_usage = []

    gpu_results = {}
    for batch_size in batch_sizes:

        gpu_results[batch_size] = run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)

    print(gpu_results)
    #     # Calculate FLOPs (floating-point operations) for the model
    #     num_flops = batch_size * num_tokens * embedding_dim * embedding_dim * num_heads
    #     flops.append(num_flops)

    #     # Calculate memory usage
    #     memory_utilization(device)
    #     memory_usage.append(torch.cuda.memory_reserved(0) / (1024 ** 3))

    # print("Flops: ", str(flops))
    # print("Memory usage: ", str(memory_usage))


    
    
    total_params = get_model_parameters(transformer)
    
    batch_size = 1
    num_tokens = 10
    embedding_dim = 256
    
    run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
