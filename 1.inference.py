import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# Preprocess the input image
def preprocess_image(image_path, image_height, image_width):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Postprocess the output image
def postprocess_image(output, image_height, image_width):
    output = np.squeeze(output)  # Remove batch dimension
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = output * 255.0
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

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
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
                
        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()
        output_d64 = np.array(output_buffer, dtype=np.float32)

        output_image = postprocess_image(output_d64, image_height, image_width)
        cv2.imwrite(output_file, output_image)
        
if __name__ == "__main__":
    image_height = 224
    image_width = 224
    infer("./model.trt", "./prediksi4.jpg", "out1.jpg")