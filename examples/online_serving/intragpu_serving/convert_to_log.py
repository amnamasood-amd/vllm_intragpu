import os
import pickle

with open("/workspace/vllm_intragpu_prev/examples/online_serving/intragpu_serving/decode_iteration.log","rb") as in_file:
    log = pickle.load(in_file)
    log_str = [",".join(x) for x in log]
    #print(log_str)
    with open("/workspace/vllm_intragpu_prev/examples/online_serving/intragpu_serving/decode_iter_data.log","w") as out_file:
        out_file.write("\n".join(log_str))

with open("/workspace/vllm_intragpu_prev/examples/online_serving/intragpu_serving/prefill_iteration.log","rb") as in_file:
    log = pickle.load(in_file)
    log_str = [",".join(x) for x in log]
    with open("/workspace/vllm_intragpu_prev/examples/online_serving/intragpu_serving/prefill_iter_data.log","w") as out_file:
        out_file.write("\n".join(log_str))