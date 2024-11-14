import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# Preprocess the input image
def preprocess_image(image_path, image_height, image_width, batch_size=5):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    # Duplicate the image to create a batch of 5
    batch_image = np.tile(image, (batch_size, 1, 1, 1))
    return batch_image


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
    print(f"Reading input image from file {input_file}")
    input_image = preprocess_image(input_file, image_height, image_width, batch_size=5)
    print(f"Input image shape after preprocessing: {input_image.shape}")  # Should print (5, 3, 416, 416)
    
    with engine.create_execution_context() as context:
        # Set explicit batch size if supported
        context.set_binding_shape(0, input_image.shape)
        
        bindings = []
        input_memory = None
        output_memories = []
        output_buffers = []

        for binding in range(engine.num_bindings):
            # Use trt.volume and binding shape to get the exact size needed
            shape = context.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print(f"Binding {binding} (Name: {engine.get_binding_name(binding)}): Shape={shape}, Size={size}, Dtype={dtype}")
            
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
                print(f"Input binding: {binding}, shape={input_image.shape}, size={input_image.nbytes}")
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                output_buffers.append(output_buffer)
                output_memories.append(output_memory)
                bindings.append(int(output_memory))
                print(f"Output binding: {binding}, shape={output_buffer.shape}, size={output_buffer.nbytes}")
                
        stream = cuda.Stream()
        try:
            # Transfer input data to the GPU
            print("Transferring input data to GPU.")
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            
            # Run inference
            print("Executing inference.")
            try:
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            except Exception as e:
                print(f"Inference execution failed: {e}")
                return
            
            # Transfer predictions back from the GPU for each output
            print("Transferring output data from GPU.")
            for output_memory, output_buffer in zip(output_memories, output_buffers):
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            
            # Synchronize the stream to ensure all operations are complete
            stream.synchronize()
            
            # Process each output tensor separately
            output_tensors = [postprocess_output(buffer, image_height, image_width) for buffer in output_buffers]
            for i, output_tensor in enumerate(output_tensors):
                print(f"Output tensor {i}:", output_tensor)
                # Save each output tensor to a separate file if needed
                np.save(f"{output_file}_output_{i}.npy", output_tensor)
                
        except cuda.Error as e:
            print(f"CUDA Error: {e}")

        
if __name__ == "__main__":
    image_height = 416
    image_width = 416
    infer("./model.trt", "./prediksi4.jpg", "out1.npy")