from datetime import datetime
import numpy as np

def save_attention_tensors(output, weights):
    dt = datetime.now()
    str_date = dt.strftime('%d%B%Y-%H%M')
    for key, tensor in {'output': output, 'weight': weights}.items():
        print(tensor.shape)
        for slice in tensor:
            print(slice.shape)
            np.savetxt(f'./attention/{key}_{str_date}.txt', slice.detach().numpy())


# def save_attention_tensors(output, weights):
#     dt = datetime.now()
#     str_date = dt.strftime('%d%B%Y-%H%M')
#     for key, tensor in {'output': output, 'weight': weights}.items():
#         txt_tensor = str(tensor.data)
#         with open(f'./attention/{key}_{str_date}.txt', 'w') as file:
#             file.write(txt_tensor)
        