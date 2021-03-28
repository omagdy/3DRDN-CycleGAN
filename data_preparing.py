
def flip_model_x(model):
    return model[::-1, :, :]

def flip_model_y(model):
    return model[:, ::-1, :]

def flip_model_z(model):
    return model[:, :, ::-1]

def get_batch_data(data, idx, BATCH_SIZE, r_x=0, r_y=0, r_z=0):
    sub_data = data[idx:idx+BATCH_SIZE]
    # Randomly flip
    for i in range(BATCH_SIZE):
        if r_x == 1:
            sub_data[i] = flip_model_x(sub_data[i])
        if r_y == 1:
            sub_data[i] = flip_model_y(sub_data[i])
        if r_z == 1:
            sub_data[i] = flip_model_z(sub_data[i])
    return sub_data

def fix_shape(data, PATCH_SIZE):
    return data.reshape(PATCH_SIZE,PATCH_SIZE,PATCH_SIZE)
