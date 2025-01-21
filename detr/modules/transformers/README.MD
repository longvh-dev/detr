# DETR Transformers Module

References the torch.nn.transformers module and provides additional:
- Positional Encoding is passed in Multi Head Attention
- Extra Linear layer at end of encoder is removed
- Decoder returns a stack of activations from all decoding layer