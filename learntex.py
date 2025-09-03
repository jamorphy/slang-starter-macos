import numpy as np
import pathlib
import slangpy as spy

device = spy.create_device(include_paths=[pathlib.Path(__file__).parent.absolute()])
module = spy.Module.load_from_file(device, "learntex.slang")

eval_network = module.run_net

b = np.random.randn(64).astype(np.float32) * 0.1
w = np.random.randn(64, 2).astype(np.float32) * 0.1
bg = np.random.randn(64).astype(np.float32) * 0.1
wg = np.random.randn(64, 2).astype(np.float32) * 0.1
layer1_b = spy.Tensor.from_numpy(device, b)
layer1_w = spy.Tensor.from_numpy(device, w)

params = {
    "layer1": {
        "b": layer1_b,
        "w": layer1_w,
        # 
        "b_grad": spy.Tensor.from_numpy(device, bg),
        "w_grad": spy.Tensor.from_numpy(device, wg)
    }
}

uv = np.array([0.5, 0.2565], dtype=np.float32)
out = eval_network(uv)

result = np.array(out)
print(result)

# rand_img = np.random.rand(128*128*4).astype(np.float32) * 0.25
# tex = device.create_texture(
#     width=128,
#     height=128,
#     format=spy.Format.rgba32_float,
#     usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
#     data=rand_img
# )

# spy.tev.show(tex, name="photo")

