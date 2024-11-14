import numpy as np
import os 
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt 
import cv2

TRT_LOGGER = trt.Logger()

def preprocess_image(image_path, image_height, image_width):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_image(output, image_height, image_width):
    output = np.squeeze(output)  # Remove batch dimension
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = output * 255.0
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def preprocess(img):
    img_data =  np.array(img).astype('float32')/float(255.0)
    return img_data

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def infer(engine_file_path, input_file, output_file):
    engine = load_engine(engine_file_path)
    print("Reading input image from file {}".format(input_file))
    input_image = preprocess_image(input_file, image_height, image_width)
    
    with engine.create_execution_context() as context:
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (1, 3, image_height, image_width))
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))
                
        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()
        output_d64 = np.array(output_buffer, dtype=np.float32)

        output_image = postprocess_image(output_d64, image_height, image_width)
        cv2.imwrite(output_file, output_image)
        
if __name__ == "__main__":
    image_height = 224
    image_width = 224
    infer("./test_static.engine", "./prediksi4.jpg", "out1.jpg")