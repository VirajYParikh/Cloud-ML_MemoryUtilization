import numpy as np

# Define the tensor shape and data type
shape = (128, 8, 256)  # Example shape (batch_size, sequence_length, input_features)
data_type = np.float32  # Example data type (float32)

# Calculate the data size
data_size = np.prod(shape) * data_type().itemsize

# Print the result
print(f"Data Size (bytes): {data_size} bytes")
