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

# Postprocess the output tensor
def postprocess_output(output, image_height, image_width):
    # Assuming the output is a tensor with shape (batch_size, num_boxes, 7)
    # where each box has 7 values: [x, y, w, h, conf, class_id, ...]
    print(f"Output shape: {output.shape}")
    output = np.reshape(output, (-1, 7))
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
        input_memory = None
        output_memory = None
        output_buffer = None

        for binding in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print(f"Binding {binding}: size={size}, dtype={dtype}")
            
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
                print(f"Input binding: {binding}, shape={input_image.shape}, size={input_image.nbytes}")
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
                print(f"Output binding: {binding}, shape={output_buffer.shape}, size={output_buffer.nbytes}")
                
        stream = cuda.Stream()
        try:
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            
            # Synchronize the stream
            stream.synchronize()
            output_d64 = np.array(output_buffer, dtype=np.float32)

            output_tensor = postprocess_output(output_d64, image_height, image_width)
            print("Output tensor:", output_tensor)
            
            # Save the output tensor to a file
            np.save(output_file, output_tensor)
        except cuda.Error as e:
            print(f"CUDA Error: {e}")
        
if __name__ == "__main__":
    image_height = 416
    image_width = 416
    infer("./model.trt", "./prediksi4.jpg", "out1.npy")