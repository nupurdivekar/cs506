import gdown
import os

def download_model():
    file_id = '1MCeY5EMv3xC6g4uvenP8JB1pcfAhvhhA'
    output_dir = os.path.join('api', 'models')
    output_path = os.path.join(output_dir, 'breast_cancer_cnn_model (1).keras')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

if __name__ == "__main__":
    download_model()