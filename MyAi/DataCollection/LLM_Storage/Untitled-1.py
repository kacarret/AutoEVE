import torch

# Example tensors
idx = torch.tensor([[39, 1549, 1141, 1, 33, 1629, 1621, 747, 1255, 11, 1621, 4, 304, 846, 2118, 106, 1340, 3652, 3589, 144, 3218, 1310, 2210, 43, 2944, 2244, 3030, 15, 1621, 4, 2767, 3217, 2270, 732, 223, 3218, 1566, 2210, 2118, 1181, 174, 3217, 1621, 4, 304, 2274, 3218, 2085, 1629, 1621, 747, 15, 474, 2884, 1621]])
idxin = torch.tensor([[39, 1549, 1141, 1, 33]])

# Convert tensors to lists for easier manipulation
idx_list = idx[0].tolist()
idxin_list = idxin[0].tolist()  # Convert the first row of idxin to a list

print(idx_list)
print(idxin_list)

# Find the first occurrence of idxin_list in idx_list
for i in range(len(idx_list) - len(idxin_list) + 1):
    if idx_list[i:i+len(idxin_list)] == idxin_list:
        # Remove the first occurrence of idxin_list from idx_list
        idx_list = idx_list[:i] + idx_list[i+len(idxin_list):]
        break
    else:
        print(f"Could not find {idxin_list} in idx_list")

# Convert the modified list back to a tensor
idxout = torch.tensor(idx_list, dtype=torch.long)

# Optional: Add an extra dimension if necessary (e.g., unsqueeze(0))
# idxout = idxout.unsqueeze(0).to(Hyperparameters.device)  # If needed

print(idxout)
