import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# Preprocess the input frame
def preprocess_frame(frame, image_height, image_width):
    frame = cv2.resize(frame, (image_width, image_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32)
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Postprocess the output tensor
def postprocess_output(output):
    # Assuming the output is a tensor with shape (batch_size, num_boxes, 7)
    # where each box has 7 values: [x, y, w, h, conf, class_id, ...]
    output = np.reshape(output, (-1, 7))
    return output

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer_video(engine_file_path, input_video, output_video):
    engine = load_engine(engine_file_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

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
                input_memory = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
                
        stream = cuda.Stream()
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                input_frame = preprocess_frame(frame, image_height, image_width)
                cuda.memcpy_htod_async(input_memory, input_frame, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                
                # Synchronize the stream
                stream.synchronize()
                output_d64 = np.array(output_buffer, dtype=np.float32)
                output_tensor = postprocess_output(output_d64)
                print("Output tensor:", output_tensor)
                
                # Here you can add code to draw bounding boxes on the frame based on output_tensor
                # For simplicity, we will just write the original frame to the output video
                out.write(frame)
                
        except cuda.Error as e:
            print(f"CUDA Error: {e}")
        finally:
            cap.release()
            out.release()
        
if __name__ == "__main__":
    image_height = 416
    image_width = 416
    infer_video("./model.trt", "./video1.MP4", "./output_video.avi")