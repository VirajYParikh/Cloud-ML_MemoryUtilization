import torch
from torch.nn import TransformerEncoderLayer
import torch.cuda.profiler as profiler
from torch.cuda import memory_allocated, memory_reserved
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
import argparse


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
    memory_utilization(device)
    output = F.log_softmax(out, dim=1)
    return output
    
    
    
    
    

def memory_utilization(device):
  if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    return round(torch.cuda.memory_allocated(0)/1024**3,1) + round(torch.cuda.memory_reserved(0)/1024**3,1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()




    num_tokens = 8
    embedding_dim = 256
    min_batch_size = 128
    max_batch_size = 1024
    num_heads = 8

    transformer, device = initialize_transformer()
    
    print(transformer)
    print("#############################Running Profiler##############################")

    batch_sizes = list(range(1, max_batch_size + 1))
    memory_usage = []

    gpu_results = []
    for batch_size in range(min_batch_size, max_batch_size + 1, 64):

        output = run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
        print("Output: ",output)
        # gpu_results.append(gpu_result)


    # print(gpu_results)

    print("#############################Model Parameters##############################")
    
    total_params = get_model_parameters(transformer)
    
    # batch_size = 1
    # num_tokens = 10
    # embedding_dim = 256
    
    # run_profiler_experiment(transformer, device, batch_size, num_tokens, embedding_dim)
