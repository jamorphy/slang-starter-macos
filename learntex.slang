// https://github.com/shader-slang/neural-shading-s25
import slangpy;

struct LayerParams<int Inputs, int Outputs>
{
  Tensor<float, 1> b;
  Tensor<float, 2> w;

  AtomicTensor<float, 1> b_grad;
  AtomicTensor<float, 2> w_grad;

  [Differentiable]
  float get_bias(int neuron)
  {
    return b.get({neuron});
  }
  [Differentiable]
  float get_weight(int neuron, int input)
  {
    return w.get({neuron, input});
  }
  [BackwardDerivativeOf(get_bias)]
  void get_bias_bwd(int neuron, float grad)
  {
    b_grad.set({neuron}, grad);
  }
  [BackwardDerivativeOf(get_weight)]
  void get_weight_bwd(int neuron, int input, float grad)
  {
    w_grad.set({neuron, input}, grad);
  }

  [Differentiable]
  float[Outputs] forward(float[Inputs] x)
  {
    float[Outputs] out;

    [MaxIters(Outputs)]
    for (int row = 0; row < Outputs; ++row)
    {
      var sum = get_bias(row);
      [ForceUnroll]
      for (int col = 0; col < Inputs; ++col) {
        sum += get_weight(row, col) * x[col];
      }
      out[row] = sum;
    }
    return out;
  }
}

struct Network
{
  LayerParams<2, 3> layer1;
  // LayerParams<128, 256> layer3;
  // LayerParams<256, 128> layer4;

  [Differentiable]
  // TODO: what's no_diff?
  float3 eval(no_diff float2 uv)
  {
    float inputs[2] = {uv.x, uv.y};

    float output0[3] = layer1.forward(inputs);

    [ForceUnroll]
    for (int i = 0; i < 3; ++i) {
      output0[i] = sigmoid(output0[i]);
    }

    return float3(output0[0], output0[1], output0[2]);
  }
}

[Differentiable]
float sigmoid(float x)
{
  return 1 / (1 + exp(-x));
}

// TODO: remove this hack
Network net;
[Differentiable]
float3 run_net(float2 uv)
{
  return net.eval(uv);
}
