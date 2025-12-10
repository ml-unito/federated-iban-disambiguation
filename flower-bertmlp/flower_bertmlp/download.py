import os
import gdown
import requests


def check_file_exists(filePath):
    """ """
    return os.path.exists(filePath)



def download_pre_trained_model():
    """ """
    
    # https://drive.google.com/file/d/1fWViT-Dbj7XmyLH4C3E6Dnrr9XJRKPjy/view?usp=sharing
    # https://drive.google.com/file/d/1ZRCdZUWU5w-oMN62rRtdL7hXdx5usbRI/view?usp=sharing
    if not check_file_exists("./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"):
        print("Downloading model pre-trained...")
        
        try:
            file_id = "1ZRCdZUWU5w-oMN62rRtdL7hXdx5usbRI"
            url = f'https://drive.google.com/uc?id={file_id}'
            output = "./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"
            gdown.download(url, output)
        
        except requests.exceptions.HTTPError as err:
            print(f'HTTP Error: {err}')
        except requests.exceptions.ConnectionError as errc:
            print(f'Error Connecting: {errc}')
        except requests.exceptions.Timeout as errt:
            print(f'Timeout Error: {errt}')
        except requests.exceptions.RequestException as errr:
            print(f'OOps: Something Else problem occurs...: {errr}')
    
    else:
        print("Model already downloaded")
