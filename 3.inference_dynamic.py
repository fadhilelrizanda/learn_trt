import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

def preprocess_frame(frame, image_height, image_width):
    # Upload the frame to the GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    # Resize on the GPU
    gpu_resized = cv2.cuda.resize(gpu_frame, (image_width, image_height))
    
    # Convert color space on the GPU (BGR to RGB)
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
    
    # Download the processed image back to the host (if TensorRT expects host memory)
    frame = gpu_rgb.download()
    
    # Normalize and prepare tensor
    frame = frame.astype(np.float32)
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    
    return frame

# Postprocess the output tensor to extract bounding boxes
def postprocess_output(output, conf_threshold=0.5):
    # Assuming the output is a tensor with shape (batch_size, num_boxes, 7)
    # where each box has 7 values: [x, y, w, h, conf, class_id, ...]
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
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
bindings = []

def infer_video(engine_file_path, input_video, output_video, batch_size, labels):
    engine = load_engine(engine_file_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0 or width == 0 or height == 0:
        print("Error: Invalid video properties.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Check if VideoWriter initialized successfully
    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    with engine.create_execution_context() as context:
        bindings, stream = [], cuda.Stream()

        # Allocate buffers
        for binding in range(engine.num_bindings):
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                bindings.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))
            else:
                bindings.append(cuda.mem_alloc(size * np.dtype(dtype).itemsize))

        try:
            frames, frame_count = [], 0
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame if dimensions mismatch
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                input_frame = preprocess_frame(frame, 416, 416)
                frames.append(input_frame)

                if len(frames) == batch_size:
                    input_batch = np.vstack(frames)
                    context.set_binding_shape(0, input_batch.shape)
                    
                    # Inference
                    cuda.memcpy_htod_async(bindings[0], input_batch, stream)
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    output = np.empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
                    cuda.memcpy_dtoh_async(output, bindings[1], stream)
                    stream.synchronize()

                    # Post-process and write frames
                    output_tensor = postprocess_output(output)
                    for f in frames:
                        f = np.transpose(f[0], (1, 2, 0))
                        f = (f * 255).astype(np.uint8)
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                        draw_boxes(f, output_tensor, labels)
                        out.write(f)

                    frames = []
                    frame_count += batch_size

            elapsed_time = time.time() - start_time
            print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({frame_count / elapsed_time:.2f} FPS)")

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
    batch_size = 1  # Increase the batch size to improve GPU utilization
    labels = load_labels("obj.names")
    infer_video("./dynamic_tsr_model.trt", "./video_1.MP4", "./output_video.avi", batch_size, labels)