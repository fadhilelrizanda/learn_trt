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

def infer_video(engine_file_path, input_video, output_video, batch_size, labels):
    engine = load_engine(engine_file_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0.0:
        print("Warning: FPS is zero. Setting to 30.")
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    print(f'Video properties: fps={fps}, size=({width}, {height})')

    # Check if VideoWriter is opened successfully
    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    with engine.create_execution_context() as context:
        # Allocate memory
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        
        # Get input and output binding indices
        input_binding_idx = engine.get_binding_index(engine.get_binding_name(0))
        output_binding_idx = engine.get_binding_index(engine.get_binding_name(1))
        input_shape = engine.get_binding_shape(input_binding_idx)
        output_shape = engine.get_binding_shape(output_binding_idx)
        input_size = trt.volume(input_shape)
        output_size = trt.volume(output_shape)
        input_dtype = trt.nptype(engine.get_binding_dtype(input_binding_idx))
        output_dtype = trt.nptype(engine.get_binding_dtype(output_binding_idx))
        # Allocate host and device buffers
        h_input = cuda.pagelocked_empty(input_size, dtype=input_dtype)
        d_input = cuda.mem_alloc(h_input.nbytes)
        h_output = cuda.pagelocked_empty(output_size, dtype=output_dtype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        bindings = [int(d_input), int(d_output)]
        
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_frame = frame.copy()

            # Preprocess the frame
            input_frame = preprocess_frame(frame, input_shape[2], input_shape[3])  # Assuming NCHW format
            np.copyto(h_input, input_frame.ravel())

            # Transfer input data to the GPU
            cuda.memcpy_htod_async(d_input, h_input, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()

            # Postprocess the output
            output_tensor = postprocess_output(h_output)

            # Draw bounding boxes on the original frame
            draw_boxes(orig_frame, output_tensor, labels)
            # Write the frame to the output video
            out.write(orig_frame)

            frame_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_actual = frame_count / elapsed_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps_actual:.2f} FPS)")

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