import math, ctypes
hip = ctypes.CDLL("libamdhip64.so")
import torch

def int_to_maskarr(mask_int, length):
    out = []
    for _ in range(length):
        out.append(mask_int & 0xFFFFFFFF)
        mask_int >>= 32
    return out
 
def stream_with_cu_mask(mask_bits):
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip.hipExtStreamCreateWithCUMask.restype  = ctypes.c_int
    hip.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
    ]
    raw_stream = ctypes.c_void_p()
    mask_arr   = (ctypes.c_uint * len(mask_bits))(*mask_bits)
    ret = hip.hipExtStreamCreateWithCUMask(
        ctypes.byref(raw_stream), len(mask_bits), mask_arr
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

def regular_stream_nonblocking():
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip.hipStreamCreateWithFlags.restype  = ctypes.c_int
    hip.hipStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    raw_stream = ctypes.c_void_p()
    ret = hip.hipStreamCreateWithFlags(
        ctypes.byref(raw_stream), 1
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

def priority_stream_nonblocking():
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip.hipStreamCreateWithPriority.restype  = ctypes.c_int
    hip.hipStreamCreateWithPriority.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    raw_stream = ctypes.c_void_p()
    ret = hip.hipStreamCreateWithPriority(
        ctypes.byref(raw_stream), 1, -1
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

def regular_stream():
    """Return torch.cuda.ExternalStream limited to the given CU mask."""
    hip.hipStreamCreate.restype  = ctypes.c_int
    hip.hipStreamCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
    ]
    raw_stream = ctypes.c_void_p()
    ret = hip.hipStreamCreate(
        ctypes.byref(raw_stream)
    )
    assert ret == 0, f"HIP err {ret} creating masked stream"
    return torch.cuda.ExternalStream(raw_stream.value)

def destroy_external_stream(es: torch.cuda.ExternalStream):
    h = ctypes.c_void_p(int(es.cuda_stream))
    ret = hip.hipStreamSynchronize(h)
    if ret != 0:
        raise RuntimeError(f"HIP err {ret} synchronizing stream before destroy")
    ret = hip.hipStreamDestroy(h)
    if ret != 0:
        raise RuntimeError(f"HIP err {ret} destroying stream")

def check_stream_flags(s):
    stream_handle=s._as_parameter_
    flags = ctypes.c_uint()
    priority = ctypes.c_int()
    hip.hipStreamGetFlags(stream_handle, ctypes.byref(flags))
    hip.hipStreamGetPriority(stream_handle, ctypes.byref(priority)) 
    print(flags.value, priority.value, s.device)

#torch.cuda.empty_cache()
torch.cuda.set_device(0) 
streams=[]

#for i in range(16):
#    streams.append(torch.cuda.Stream())
    #print(streams[i].cuda_stream)
#streams.append(torch.cuda.Stream())
for i in range(32):
    streams.append(regular_stream_nonblocking())
for i in range(32):
    streams.append(priority_stream_nonblocking())
#check_stream_flags(streams[0])
#check_stream_flags(priority_stream_nonblocking())
#print(streams[0].priority)
#streams.append(regular_stream())
for i in range(1,20):
    cu_mask_int=(1<<32*i)-1
    #print(cu_mask_int)
    cu_mask=int_to_maskarr(cu_mask_int,10)
    #print(cu_mask)
    streams.append(stream_with_cu_mask(cu_mask))
    #check_stream_flags(streams[i])
    #streams.append(torch.cuda.Stream())

# x=torch.rand(1024,1024).to("cuda:0")
# y=torch.rand(1024,1024).to("cuda:0")
# z = []
# #print(x)
# for s in streams:
#     with torch.cuda.stream(s):
#         z.append(torch.matmul(x,y))

# print(z)
