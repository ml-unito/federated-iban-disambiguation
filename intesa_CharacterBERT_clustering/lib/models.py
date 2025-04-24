import torch.nn as nn
from fluke import nets
from character_bert_model.modeling.character_bert import CharacterBertModel


class NERClassifier_D(nn.Module):
    def __init__(self):
        super(NERClassifier_D, self).__init__()

        num_labels = 2
        self.dropout = nn.Dropout(0.2)
        self.hidden = nn.Linear(768, 100)
        self.classifier = nn.Linear(100, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, outputs):
        pooled_output = self.dropout(
            outputs[:, 0, :]
        )  # Take the first token's embedding ([CLS])

        hidden_logits = self.relu(self.hidden(pooled_output))
        logits = self.classifier(hidden_logits)

        # x = self.sigmoid(logits)

        return logits

class NERClassifier_E(nn.Module):
    def __init__(self):
        super(NERClassifier_E, self).__init__()
        self.character_bert = CharacterBertModel.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
        
    def forward(self, x):
        # Forward pass through the CharacterBert model
        outputs = self.character_bert(x)[0]
        return outputs

class NERClassifier(nets.EncoderHeadNet):
    def __init__(self):
        """
        NERClassifier is a neural network model for Named Entity Recognition (NER) tasks.
        """

        super(NERClassifier, self).__init__(
            NERClassifier_E(),
            NERClassifier_D()
        )
