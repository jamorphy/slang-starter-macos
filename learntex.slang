// https://github.com/shader-slang/neural-shading-s25
import slangpy;

[Differentiable]
float3 layer_forward(
    Tensor<float, 1> b,
    Tensor<float, 2> w,
    float2 x)
{
    float3 out;

    [MaxIters(3)]
    for (int row = 0; row < 3; ++row)
    {
        var sum = b.get({row});
        [ForceUnroll]
        for (int col = 0; col < 2; ++col) {
            sum += w.get({row, col}) * x[col];
        }
        out[row] = sum;
    }
    return out;
}

[Differentiable]
float3 net_forward(
    Tensor<float, 2> w,
    Tensor<float, 1> b,
    float2 uv)
{
    float2 inputs = float2(uv.x, uv.y);
    float3 hidden = layer_forward(b, w, inputs);

    [ForceUnroll]
    for (int i = 0; i < 3; ++i) {
        hidden[i] = sigmoid(hidden[i]);
    }

    return float3(hidden[0], hidden[1], hidden[2]);
}

[Differentiable]
float sigmoid(float x)
{
  return 1 / (1 + exp(-x));
}
