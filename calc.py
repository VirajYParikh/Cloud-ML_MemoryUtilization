import re

# Initialize variables to store the sums of each metric
fadd_sum = 0
ffma_sum = 0
fmul_sum = 0

# Read the text file
with open('your_file.txt', 'r') as file:
    text = file.read()

# Define a regular expression pattern to match the metric lines
pattern = r'smsp__sass_thread_inst_executed_op_(\w+)_pred_on.sum\s+inst\s+(\d+)'

# Find all matches in the text
matches = re.findall(pattern, text)

# Iterate through the matches and update the sums
for match in matches:
    metric_name, metric_value = match
    metric_value = int(metric_value)
    if metric_name == 'fadd':
        fadd_sum += metric_value
    elif metric_name == 'ffma':
        ffma_sum += metric_value
    elif metric_name == 'fmul':
        fmul_sum += metric_value

# Calculate the total sum
total_sum = fadd_sum + ffma_sum + fmul_sum

# Print the sums
print(f"Total fadd_sum: {fadd_sum}")
print(f"Total ffma_sum: {ffma_sum}")
print(f"Total fmul_sum: {fmul_sum}")
print(f"Total Sum of all metrics: {total_sum}")
