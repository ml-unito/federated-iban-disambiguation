from lib.CharacterBertForClassification import *

class CharacterBertForEmbeddingsGeneration(CharacterBertForClassification):
    def __init__(self):
        super(CharacterBertForEmbeddingsGeneration, self).__init__(num_labels=1)

    def forward(self, input_ids):
        outputs = self.character_bert(input_ids)[0]       # Use the last hidden states
        return self.dropout(outputs[:, 0, :])           # Take the first token's embedding ([CLS])
