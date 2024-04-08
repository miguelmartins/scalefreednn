import os

def get_cuda_device_environ():
    try:
        cuda_env = os.environ['CUDA_VISIBLE_DEVICES']
        return str(cuda_env)
    except KeyError as e:
        return 'NA'