import onnx

onnx_model = onnx.load("/home/lccjh/share/work/TRT/TensorRT-LLM/examples/bloom/bloom/560M/trt_engines/fp16/1-gpu/test.onnx")
for node in onnx_model.graph.node:
    print(node)