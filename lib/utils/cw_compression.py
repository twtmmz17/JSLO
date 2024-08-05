import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def getSimMatrix(values, sim, pruned_idx):
    """
        Compute similaritys among input channel with channel according weight {values[i]} and similarity standard {sim}
    Args:
        values: weight value of a layer
        sim: similarity among input channels(max is c, [:,i,:,:]),
        pruned_idx: indice of the pruned channel
    Returns:
        1 X C 2d array
    """
    sims = []
    x = values[:, pruned_idx].reshape(1, -1)
    mask = np.ones(values.size(1), bool)
    mask[pruned_idx] = False
    y = values[mask].reshape(values.size(1) - 1, -1)
    if sim == 'cos':
        sims = cosine_similarity(x, y)
        sims = 1.0 - sims
    return sims;

def channel_wise_comresspion(layer_weight, layer_quantify_bits, pruned_idx, sim='cos', sim_threshold=0.6):
    """
    channel-wise quantization at each layer by input channel index {pruned_idx}
    1 channel-wise compression = 1 * prune + k * quantization (k in [0,c], c is the number of input channels)
    Args:
        layer_weight: weight of layer in shape [n,c, w, h]
        layer_quantify_bits: channel-wise quantization plan in shape [1 x c]
        pruned_idx: index of the pruned channel, scala
        preserved_mask: show whether preserved a specific channel, boolean array [1 x c]
        sim: approach to define similarity among the pruned channel and the rest channel, string in type ['JS

    Returns:
        compressed_layer_weights: weight after compression
        layer_quantify_bits:
    """
    # todo: 0. init
    n, c = layer_weight.size(0), layer_weight.size(1)
    similarities = getSimMatrix(layer_weight, sim, pruned_idx)
    # todo: 1. (pasudo) prune the [channel_ind]-th channels
        #todo: add update weight here
    layer_quantify_bits[pruned_idx] = 0
    sims = similarities[pruned_idx]
    compress_channels = sims >= sim_threshold
    #todo: 2. quantify the [compress_channels] according to their similarity
    #todo: here, a channel may be compressed for multiple times
    for ind in compress_channels:
        # paseudo quantization
        if layer_quantify_bits[ind] == 0: # this channel has been pruned
            continue
        if layer_quantify_bits[ind] == 8: # first selected to quantify
            # the more {ind}-th like the pruned_idx, the more it should compressed
            layer_quantify_bits[ind] = np.ceil((1 - sims[ind]) * 8)
        else:
            layer_quantify_bits[ind] = layer_quantify_bits[ind] - 1

    return layer_quantify_bits;