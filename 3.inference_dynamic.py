import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

# Preprocess the input frame
def preprocess_frame(frame, image_height, image_width):
    frame = cv2.resize(frame, (image_width, image_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            shape = engine.get_binding_shape(binding)
            print(f"Binding {binding} shape: {shape}")
            if shape[0] == -1:
                shape[0] = batch_size  # Set dynamic batch size
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print(f"Binding {binding}: size={size}, dtype={dtype}")
            
            if size < 0:
                raise ValueError(f"Invalid size {size} for binding {binding}")

            if engine.binding_is_input(binding):
                input_memory = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
                
        stream = cuda.Stream()
        try:
            frames = []
            start_time = time.time()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                input_frame = preprocess_frame(frame, image_height, image_width)
                frames.append(input_frame)
                
                if len(frames) == batch_size:
                    input_batch = np.vstack(frames)
                    input_batch = np.ascontiguousarray(input_batch)  # Ensure the array is contiguous
                    print(f"Input batch shape: {input_batch.shape}")
                    cuda.memcpy_htod_async(input_memory, input_batch, stream)
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                    
                    # Synchronize the stream
                    stream.synchronize()
                    output_d64 = np.array(output_buffer, dtype=np.float32)
                    print(f"Output buffer shape: {output_d64.shape}")
                    output_tensor = postprocess_output(output_d64)
                    print("Output tensor:", output_tensor)
                    
                    # Draw bounding boxes on the frames
                    for f in frames:
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
        
if __name__ == "__main__":
    image_height = 416
    image_width = 416
    batch_size = 1  # Set the batch size to match your model's expected batch size
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    infer_video("./dynamic_tsr_model.trt", "./video_1.MP4", "./output_video.avi", batch_size, labels)