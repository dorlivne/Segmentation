from loss import create_weights


def extract_weight_map(queue_seg, queue_weightmap):
    seg = queue_seg.get()
    queue_weightmap.put(create_weights(seg=seg))
    extract_weight_map(queue_seg, queue_weightmap)


# TODO implement this function
def Jaccard_Index(predictions, seg):
    return 0

