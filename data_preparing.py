from sklearn.model_selection import train_test_split

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

def data_pre_processing(lr_data, hr_data, BATCH_SIZE):
    lr_train, lr_temp, hr_train, hr_temp = train_test_split(lr_data, hr_data, test_size=1-0.5, random_state=42)
    lr_validation, lr_test, hr_validation, hr_test = train_test_split(lr_temp, hr_temp, test_size=0.1, random_state=42)
    data_splits = [[hr_train, lr_train], [hr_validation, lr_validation], [hr_test, lr_test]]
    clipped_data_splits = []
    for i in range(len(data_splits)):
        data_split = data_splits[i]
        hr_data = data_split[0]
        lr_data = data_split[1]
        n_data  = hr_data.shape[0]
        if n_data%BATCH_SIZE != 0:
            n_data = n_data - n_data%BATCH_SIZE
            lr_data = lr_data[0:n_data]
            hr_data = hr_data[0:n_data]
        clipped_data_splits.append([hr_data, lr_data])
    return clipped_data_splits
