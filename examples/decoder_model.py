import torch
import torch.nn as nn
from fms.models import get_model
print("start")
model = get_model("simple_llama", "test", data_type=torch.float16, fused_weights=False).to("spyre")
print("weights moved: ", model.head.weight)

embedding = nn.Embedding(
    model.config.src_vocab_size, model.config.emb_dim, model.config.pad_id, dtype=torch.float16
)
compiled_model = torch.compile(model)
with torch.no_grad():
    test_input = torch.randint(0, 32000, (1, 128), dtype=torch.int64)
    print("starting inference")
    embedded_x_in = embedding(test_input)
    embedded_x_in_spyre = embedded_x_in.to("spyre")
    print("embedding moved: ", embedded_x_in)
    print(model(embedded_x_in_spyre))