import os, requests, zipfile


download_url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
model_file = 'tensorflow_inception_graph.pb'


def download_model(data_dir):
    download_path = get_zip_file_name(data_dir)
    response = requests.get(download_url)
    with open(download_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    return get_model_file_name(data_dir)


def get_model_file_name(data_dir):
    return os.path.join(data_dir, model_file)


def get_zip_file_name(data_dir):
    base = os.path.basename(download_url)
    download_path = os.path.join(data_dir, base)
    return download_path


def download_model_if_not_exists(data_dir):
    zip_file_path = get_zip_file_name(data_dir)
    if not os.path.isfile(zip_file_path):
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        return download_model(data_dir)
    return get_model_file_name(data_dir)