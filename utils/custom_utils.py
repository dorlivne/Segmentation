from loss import create_weights


def extract_weight_map(queue_seg, queue_weightmap):
    while True:
        seg = queue_seg.get()
        try:
            if not seg:
                print("utility process exiting...")
                return
        except Exception:
            pass
        queue_weightmap.put(create_weights(seg=seg))


# TODO implement this function
def Jaccard_Index(predictions, seg):
    return 0

