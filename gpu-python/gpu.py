import tensorflow as tf

print(f"Tensorflow Version: {tf. __version__}")

print(os.system("uname -a"))
print(os.system('nvcc --version'))

gpus = tf.config.list_physical_devices(device_type='GPU')
print(f"GPUs: {gpus}",)

print(f"GPUs Available (is_gpu_available): {tf.test.is_gpu_available()}")
print(f"GPUs Available (is_built_with_cuda): {tf.test.is_built_with_cuda()}")
print(f"GPUs Available (gpu_device_name): {tf.test.gpu_device_name()}")
print(f"GPUs Available (list_physical_devices): {tf.config.list_physical_devices('GPU')}")

if tf.test.is_gpu_available():
    print('GPU is being used.')
else:
    print('GPU is not being used.')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Available GPU(s): ", len(gpus))
else:
    print("No GPUs available")
    