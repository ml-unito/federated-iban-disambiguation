import os
import gdown
import requests


def check_file_exists(filePath):
    """ """
    return os.path.exists(filePath)




def download_fine_tuned_model():
    """ """
    
    if not check_file_exists("./character_bert_model/pretrained-models/general_character_bert/pytorch_model.bin"):
        print("Downloading model fine-tuned...")
        
        try:
            file_id = "1VygI2J7dFEazV8J2Iekk3SrqutSF8CTS"
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
