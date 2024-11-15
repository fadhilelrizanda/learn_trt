import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

# Preprocess the frame entirely on the GPU
def preprocess_frame_cuda(frame, image_height, image_width):
    # Upload the frame to the GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    # Resize and convert color on the GPU (BGR to RGB)
    gpu_resized = cv2.cuda.resize(gpu_frame, (image_width, image_height))
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)

    # Create an output GpuMat for the normalized frame with CV_32F type
    gpu_normalized = cv2.cuda_GpuMat(gpu_rgb.size(), cv2.CV_32F)

    # Normalize by dividing by 255 using alpha for scaling
    gpu_rgb.convertTo(gpu_normalized, alpha=1/255.0)

    # Split channels for CHW format
    gpu_channels = cv2.cuda.split(gpu_normalized)
    return cv2.cuda.merge(gpu_channels)  # Return in CHW format as a single GpuMat

# Postprocess the output tensor to extract bounding boxes
def postprocess_output(output, conf_threshold=0.5):
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

# Load TensorRT engine
def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Infer video using TensorRT
def infer_video(engine_file_path, input_video, output_video, batch_size, labels):
    engine = load_engine(engine_file_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Prepare memory allocations and CUDA stream
    with engine.create_execution_context() as context:
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in range(engine.num_bindings):
            shape = engine.get_binding_shape(binding)
            if shape[0] == -1:
                shape[0] = batch_size  # Set dynamic batch size
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            if engine.binding_is_input(binding):
                host_inputs.append(np.empty(size, dtype=dtype))
                cuda_inputs.append(cuda.mem_alloc(host_inputs[-1].nbytes))
                bindings.append(int(cuda_inputs[-1]))
            else:
                host_outputs.append(np.empty(size, dtype=dtype))
                cuda_outputs.append(cuda.mem_alloc(host_outputs[-1].nbytes))
                bindings.append(int(cuda_outputs[-1]))

        stream = cuda.Stream()
        
        # Warm-up phase
        for _ in range(10):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

        try:
            frames = []
            start_time = time.time()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                gpu_preprocessed_frame = preprocess_frame_cuda(frame, image_height, image_width)
                frames.append(gpu_preprocessed_frame.download())

                if len(frames) == batch_size:
                    input_batch = np.vstack(frames)
                    input_batch = np.ascontiguousarray(input_batch)  # Ensure contiguous memory

                    # Set input shape and copy input data to GPU
                    context.set_binding_shape(0, input_batch.shape)
                    cuda.memcpy_htod_async(cuda_inputs[0], input_batch, stream)

                    # Perform inference asynchronously
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

                    # Copy output back to host memory asynchronously
                    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

                    # Synchronize stream
                    stream.synchronize()

                    # Post-process and draw bounding boxes
                    output_d64 = np.array(host_outputs[0], dtype=np.float32)
                    output_tensor = postprocess_output(output_d64)
                    for f in frames:
                        f = np.transpose(f[0], (1, 2, 0))  # CHW to HWC
                        f = (f * 255).astype(np.uint8)
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                        draw_boxes(f, output_tensor, labels)
                        out.write(f)

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

# Load labels from file
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

if __name__ == "__main__":
    image_height = 416
    image_width = 416
    batch_size = 4  # Increase batch size to better utilize GPU
    labels = load_labels("obj.names")
    infer_video("./dynamic_tsr_model.trt", "./video_1.MP4", "./output_video.avi", batch_size, labels)
