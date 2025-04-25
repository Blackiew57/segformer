from transformers import SegformerForSemanticSegmentation

def create_model():
    id2label = {0: 'background', 1: 'tumor'}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id)
    
    return model