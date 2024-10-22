import logging
from ..lib.CharacterBertForEmbeddingsGeneration import *
from ..lib.download import download_fine_tuned_model
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def create_model():
    """ """

    # load model
    model = CharacterBertForEmbeddingsGeneration()
    model_path = "./embedding_generator/character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"
    try: model.load_state_dict(model_path)
    except: model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
    

# Load Custom model
download_fine_tuned_model()
model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
characterBERTmodel = create_model()
characterBERTmodel = characterBERTmodel.to(model_device)
indexer = CharacterIndexer()




def create_characterBERT_embeddings(padded_tensor):
    """ create embeddings using the characterBERT fine-Tuned model """
    
    embeddings = characterBERTmodel(padded_tensor)
    return embeddings.to(model_device)



def create_characterBERT_padded_tensors(words):
    """ create embeddings using the characterBERT fine-Tuned model """
    
    padded_tensor = indexer.as_padded_tensor(words)
    return padded_tensor.to(model_device)



def create_characterBERT_embeddings_old(words):
    """ create embeddings using the characterBERT fine-Tuned model """
    
    embeddings = characterBERTmodel(indexer.as_padded_tensor(words).to(model_device))
    return embeddings.to(model_device)