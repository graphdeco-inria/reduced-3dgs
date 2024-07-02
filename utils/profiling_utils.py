import torch
try:
    import nvtx
    NVTX_FOUND = True
except ImportError:
    print("NVTX not found: not recording profiling information")
    NVTX_FOUND = False

recordTime = False

class SyncedNVTX:
    def __init__(self, message, color, domain):
        self.message = message
        self.color = color
        self.domain = domain

    def __enter__(self):
        if NVTX_FOUND and recordTime:
            torch.cuda.synchronize()
            self.range = nvtx.start_range(message=self.message, color=self.color, domain=self.domain)
        return self
    
    def enter(self):
        if NVTX_FOUND and recordTime:
            torch.cuda.synchronize()
            self.range = nvtx.start_range(message=self.message, color=self.color, domain=self.domain)
    
    def __exit__(self, type=None, value=None, traceback=None):
        if NVTX_FOUND and recordTime:
            torch.cuda.synchronize()
            nvtx.end_range(self.range)