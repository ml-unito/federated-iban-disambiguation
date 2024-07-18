import logging
from lib.CharacterBertForEmbeddingsGeneration import *
from lib.download import download_fine_tuned_model
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def create_model():
    """ """

    # load model
    model = CharacterBertForEmbeddingsGeneration()
    model_path = "./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"
    try: model.load_state_dict(model_path)
    except: model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
    

# Load Custom model
download_fine_tuned_model()
indexer = CharacterIndexer()
characterBERTmodel = create_model()


def create_characterBERT_embeddings(string):
    """ create embeddings using the characterBERT fine-Tuned model """
    
    input_tensors = indexer.as_padded_tensor([string])
    return characterBERTmodel(input_tensors)
