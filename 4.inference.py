import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

def preprocess_frame_to_gpu(frame, image_height, image_width, input_buffer):
    # Upload the frame to the GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Resize on GPU
    gpu_resized = cv2.cuda.resize(gpu_frame, (image_width, image_height))

    # Convert color space on GPU (BGR to RGB)
    gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)

    # Normalize on GPU (divide by 255)
    gpu_normalized = cv2.cuda.divideWithScalar(gpu_rgb, 255.0)

    # Convert to CHW format for TensorRT
    gpu_chw = cv2.cuda_GpuMat()
    gpu_chw.create((3, image_height * image_width), cv2.CV_32F)
    cv2.cuda.split(gpu_normalized, gpu_chw)  # Split into CHW format

    # Copy preprocessed data to TensorRT input buffer
    cuda.memcpy_htod_async(input_buffer, gpu_chw.download())

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
    # Load TensorRT engine
    engine = load_engine(engine_file_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    with engine.create_execution_context() as context:
        # Set input dimensions explicitly
        context.set_binding_shape(0, (batch_size, 3, image_height, image_width))
        
        # Allocate memory for TensorRT buffers
        input_shape = tuple(context.get_binding_shape(0))
        input_size = int(np.prod(input_shape) * np.dtype(np.float32).itemsize)  # Ensure Python int
        output_shape = tuple(context.get_binding_shape(1))
        output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)  # Ensure Python int

        input_buffer = cuda.mem_alloc(input_size)
        output_buffer = cuda.mem_alloc(output_size)
        bindings = [int(input_buffer), int(output_buffer)]

        # CUDA streams
        input_stream = cuda.Stream()
        output_stream = cuda.Stream()

        try:
            start_time = time.time()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess frame directly on GPU
                preprocess_frame_to_gpu(frame, image_height, image_width, input_buffer)

                # Run inference
                context.execute_v2(bindings=bindings)

                # Transfer output back to CPU
                output_host = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output_host, output_buffer, output_stream)
                cuda.Stream().synchronize()

                # Postprocess and draw bounding boxes
                output_tensor = postprocess_output(output_host)
                if output_video:
                    frame = draw_boxes(frame, output_tensor, labels)
                    out.write(frame)

                frame_count += 1

            elapsed_time = time.time() - start_time
            print(f"Processed {frame_count} frames in {elapsed_time:.2f}s ({frame_count / elapsed_time:.2f} FPS)")

        except cuda.Error as e:
            print(f"CUDA Error: {e}")
        finally:
            cap.release()
            if output_video:
                out.release()

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

if __name__ == "__main__":
    # Parameters
    image_height = 416
    image_width = 416
    batch_size = 1
    labels = load_labels("obj.names")

    # Run inference
    infer_video(
        engine_file_path="./dynamic_tsr_model.trt",
        input_video="./video_1.MP4",
        output_video="./output_video.avi",
        batch_size=batch_size,
        labels=labels,
    )
