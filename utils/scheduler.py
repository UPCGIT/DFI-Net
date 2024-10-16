import torch.optim as optim


def load_scheduler(model_name, model):
    optimizer, scheduler = None, None
    
    if model_name == 'm3ddcnn':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = None
    elif model_name == 'NEWdlst':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        scheduler = None

    return optimizer, scheduler


