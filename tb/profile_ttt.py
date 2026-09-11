import torch
import torch.nn as nn

from zipmap.layers.ttt import FastWeightGluMLPMultihead

def main():
    DEVICE = "cuda:0"
    DTYPE = torch.float32
    
    BATCH_SIZE = 1
    SEQ_LEN = 2048
    DIM = 768
    HEAD_DIM = 64
    
    print(f"Profiling with Tensor Shape: [Batch={BATCH_SIZE}, Seq={SEQ_LEN}, Dim={DIM}]")

    print("Initializing modules...")

    ### Standard transformer block
    transformer_block = nn.TransformerEncoderLayer(
        d_model=DIM, 
        nhead=DIM // HEAD_DIM, 
        dim_feedforward=DIM * 4, 
        batch_first=True, 
        dtype=DTYPE, 
        device=DEVICE
    )

    ### Baseline model
    ttt_block = FastWeightGluMLPMultihead(
        dim=DIM, 
        head_dim=HEAD_DIM, 
        muon_update_steps=5,
        use_fused_kernels=False  
    ).to(DEVICE).to(DTYPE)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    

    info = {
        "ttt_op_order": [[0, -1, True, True]]
    }


    print("Warming up GPU pipelines...")
    for _ in range(5):
        out_t = transformer_block(x)
        out_t.sum().backward()
        x.grad = None
        
        out_z, _ = ttt_block(x, info=info)
        out_z.sum().backward()
        x.grad = None
        
    torch.cuda.synchronize()


    print("Running Profiling...")

    torch.cuda.nvtx.range_push("Standard_Transformer")
    out_t = transformer_block(x)
    out_t.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    x.grad = None


    torch.cuda.nvtx.range_push("ZipMap_TTT_LaCT")
    out_z, _ = ttt_block(x, info=info)
    out_z.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    x.grad = None

    print("Profiling Script Executed Successfully.")

if __name__ == "__main__":
    main()
