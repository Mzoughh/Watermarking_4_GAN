from Attacks.pruning import prune_model_l1_unstructured 
from Attacks.quantization import quantization
from Attacks.noise import adding_noise_global

def attacks(net,TypeAttack,attackparameters):
    '''
    Apply a modification based on the ID and parameters
    :param TypeAttack: ID of the modification
    :param net: network to be altered
    :param parameters: parameters of the modification
    :return: altered NN
    '''
    if TypeAttack=="l1pruning":
        return prune_model_l1_unstructured(net, attackparameters["proportion"])

    elif TypeAttack=="quantization":
        return quantization(net,attackparameters["bits"])
    
    elif TypeAttack=="noise":
        return adding_noise_global(net,attackparameters["power"])    
    else:
        print("NotImplemented")
        return net
'''
    attackParameter = {'name': "all", "std":.1}
    network = attacks(network, "noise", attackParameter)
'''