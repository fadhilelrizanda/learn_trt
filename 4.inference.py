import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

def preprocess_frame_gpu(frame, image_height, image_width):
    # Upload the frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    # Resize on the GPU
    gpu_resized = cv2.cuda.resize(gpu_frame, (image_width, image_height))
    
    # Convert color space on the GPU (BGR to RGB)
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize on the GPU (divide by 255)
    gpu_normalized = cv2.cuda.divide(gpu_rgb, 255.0)
    
    # Rearrange dimensions (HWC -> CHW)
    gpu_transposed = cv2.cuda_GpuMat(image_height, image_width, cv2.CV_32FC3)
    cv2.cuda.transpose(gpu_normalized, gpu_transposed)
    
    return gpu_transposed  # GPU tensor stays on the GPU

# Postprocess the output tensor to extract bounding boxes
def postprocess_output(output, conf_threshold=0.5):
    # Assuming the output is a tensor with shape (batch_size, num_boxes, 7)
    output = np.reshape(output, (-1, 7))
    boxes = []
    for detection in output:
        x, y, w, h, conf, class_id = detection[:6]
        if conf > conf_threshold:
            boxes.append((int(x), int(y), int(w), int(h), conf, int(class_id)))
    return boxes

# Draw bounding boxes on the frame
def draw_boxes(frame, boxes, labels):
    for (x, y, w, h, conf, class_id) in boxes:
        label = f"{labels[class_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Initialize variables outside the function
host_outputs = []
cuda_outputs = []
bindings = []

def infer_video(engine_file_path, input_video, output_video, batch_size, labels):
    global host_outputs, cuda_outputs, bindings

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
    
    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    with engine.create_execution_context() as context:
        for binding in range(engine.num_bindings):
            shape = engine.get_binding_shape(binding)
            if shape[0] == -1:
                shape[0] = batch_size  # Set dynamic batch size
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            if engine.binding_is_input(binding):
                cuda_inputs = []
                bindings.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
            else:
                host_outputs.append(np.empty(size, dtype=dtype))
                cuda_outputs.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
                bindings.append(int(cuda_outputs[-1]))
                
        stream = cuda.Stream()
        try:
            frames = []
            start_time = time.time()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Ensure frame dimensions match VideoWriter dimensions
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                gpu_frame = preprocess_frame_gpu(frame, image_height, image_width)
                frames.append(gpu_frame)

                if len(frames) == batch_size:
                    for i, gpu_frame in enumerate(frames):
                        cuda.memcpy_htod_async(bindings[0], gpu_frame.cudaPtr(), stream)

                    # Set input shape explicitly
                    context.set_binding_shape(0, (batch_size, 3, image_height, image_width))

                    # Execute inference
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

                    # Retrieve and process output
                    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
                    stream.synchronize()

                    output_tensor = postprocess_output(host_outputs[0])

                    # Draw bounding boxes and write frames to the video
                    for f in frames:
                        f_host = f.download()
                        f_host = (f_host * 255).astype(np.uint8)
                        f_host = cv2.cvtColor(f_host, cv2.COLOR_RGB2BGR)
                        f_host = draw_boxes(f_host, output_tensor, labels)
                        out.write(f_host)  # Write to the video file
                    
                    frames = []
                    frame_count += batch_size
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
                
        except cuda.Error as e:
            print(f"CUDA Error: {e}")
        finally:
            cap.release()
            out.release()

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels
        
if __name__ == "__main__":
    image_height = 416
    image_width = 416
    batch_size = 4  # Increase the batch size to improve GPU utilization
    labels = load_labels("obj.names")
    infer_video("./dynamic_tsr_model.trt", "./video_1.MP4", "./output_video.avi", batch_size, labels)
