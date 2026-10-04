import torch
from models.pidnet import get_pred_model

def get_model_summary(model_name='pidnet_m', num_classes=19, input_size=(1, 3, 1024, 2048)):
    print(f"--- Model Summary for {model_name} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = get_pred_model(name=model_name, num_classes=num_classes)
    model.eval()
    model.to(device)
    
    # 1. Calculate conventional parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params / 1e6:.4f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.4f} M")
    
    # 1.5 Try to print model architecture using `torchinfo` or `torchsummary`
    # NOTE: torchsummary does NOT support models with tuple-input submodules (e.g. FFC).
    #       torchinfo does — use `pip install torchinfo` for full support.
    try:
        from torchinfo import summary
        print("\n[torchinfo] Model Architecture Details:")
        summary(model, input_size=input_size, device=device.type)
    except ImportError:
        try:
            from torchsummary import summary
            print("\n[torchsummary] Model Architecture Details:")
            summary(model, input_size=input_size[1:], device=device.type)
        except ImportError:
            print("\n[Optional] Install 'torchinfo' for detailed architecture: pip install torchinfo")
        except AttributeError:
            print("\n[torchsummary] Failed: this model contains FFC layers with tuple inputs,")
            print("               which torchsummary does not support.")
            print("               Install torchinfo instead: pip install torchinfo")
    
    # 2. Try to calculate FLOPs and Parameters using `thop`
    try:
        from thop import profile
        from thop import clever_format
        
        dummy_input = torch.randn(input_size).to(device)
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
        
        print(f"\n[thop] Input shape: {input_size}")
        print(f"[thop] MACs (approx FLOPs): {macs_fmt}")
        print(f"[thop] Parameters: {params_fmt}")
        
    except ImportError:
        print("\n[Optional] 'thop' library not found. Run `pip install thop` if you want to calculate MACs/FLOPs.")
        
    # 3. Try to calculate complexity using `ptflops`
    try:
        from ptflops import get_model_complexity_info
        
        # ptflops input shape should not include the batch size dimension
        ptflops_input = input_size[1:] 
        macs, params = get_model_complexity_info(model, ptflops_input, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        
        print(f"\n[ptflops] Input shape: {ptflops_input}")
        print(f"[ptflops] Computational complexity (MACs): {macs}")
        print(f"[ptflops] Number of parameters: {params}")
        
    except ImportError:
        pass
        
if __name__ == '__main__':
    # You can change the model name to 'pidnet_s', 'pidnet_m', or 'pidnet_l'
    get_model_summary(model_name='pidnet_s', num_classes=19, input_size=(1, 3, 1024, 2048))
    # print("\n")
    # get_model_summary(model_name='pidnet_m', num_classes=19, input_size=(1, 3, 1024, 2048))
    # print("\n")
    # get_model_summary(model_name='pidnet_l', num_classes=19, input_size=(1, 3, 1024, 2048))
