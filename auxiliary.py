import pickle,os,torch
import numpy as np

#Should be shifted to some other locations
def load_relation(COMMON_DATA_DIR):
    """
        Function used once to load the relations dictionary
        (which keeps the log of IDified relations, their uri and other things.)

    :param relation_file: str
    :return: dict
    """
    relations = pickle.load(open(os.path.join(COMMON_DATA_DIR, 'relations.pickle')))
    inverse_relations = {}
    for key in relations:
        value = relations[key]
        new_key = value[0]
        value[0] = key
        inverse_relations[new_key] = value

    return inverse_relations


def save_location(problem, model_name, dataset):
    '''
            Location - data/models/problem/model_name/dataset/0X
            problem - core_chain
                    -intent
                    -rdf
                    -type_existence
            model_name - cnn_dense_dense ; pointwise_cnn_dense_dense ....

            dataset -
            return a dir data/models/problem/model_name/dataset/0X
    '''
    # Check if the path exists or not. If not create one.
    assert (problem in ['core_chain', 'intent', 'rdf', 'type_existence'])
    assert (dataset in ['qald', 'lcquad', 'transfer-a', 'transfer-b', 'transfer-c'])

    path = 'data/models/' + str(problem) + '/' + str(model_name) + '/' + str(dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    dir_name = [int(name) for name in os.listdir(path + '/')]
    if not dir_name:
        new_path_dir = path + '/' + str(0)
        os.mkdir(new_path_dir)
    else:
        dir_name = max(dir_name)
        new_model = dir_name + 1
        new_path_dir = path + '/' + str(new_model)
        os.mkdir(new_path_dir)
    return new_path_dir


# Function to save the model
def save_model(loc, model, model_name='model.torch', epochs=0, optimizer=None, accuracy=0):
    """
        Input:
            loc: str of the folder where the models are to be saved - data/models/core_chain/cnn_dense_dense/lcquad/5'
            models: dict of 'model_name': model_object
            epochs, optimizers are int, torch.optims (discarded right now).
    """

    state = {
        'epoch': epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'accuracy': accuracy
    }
    loc = loc + '/' + model_name
    print("model with accuracy ", accuracy, "stored at", loc)
    torch.save(state, loc)


def validation_accuracy(valid_questions, valid_pos_paths, valid_neg_paths, modeler, model,device):
    precision = []
    with torch.no_grad():
        for i in range(len(valid_questions)):
            question = np.repeat(valid_questions[i].reshape(1, -1), len(valid_neg_paths[i]) + 1,
                                 axis=0)  # +1 for positive path
            paths = np.vstack((valid_pos_paths[i].reshape(1, -1), valid_neg_paths[i]))


            question = torch.tensor(question, dtype=torch.long, device=device)
            paths = torch.tensor(paths, dtype=torch.long, device=device)
            score = modeler.predict(question, paths, model, device)
            arg_max = torch.argmax(score)
            if arg_max.item() == 0:  # 0 is the positive path index
                precision.append(1)
            else:
                precision.append(0)
    return sum(precision) * 1.0 / len(precision)