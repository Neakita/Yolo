using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

public sealed class RawOutput : IDisposable
{
	internal static RawOutput Create(OrtIoBinding binding, TensorInfo tensorInfo)
	{
		var output0Info = tensorInfo.Output0;
		var output1Info = tensorInfo.Output1;
		var output0TensorOwner = DenseTensorOwner<float>.Allocate(output0Info);
		BindOutput(tensorInfo.Output0Name, binding, output0TensorOwner.Tensor, output0Info.Dimensions64);
		if (output1Info == null)
			return new RawOutput(output0TensorOwner);
		var output1TensorOwner = DenseTensorOwner<float>.Allocate(output1Info.Value);
		Guard.IsNotNull(tensorInfo.Output1Name);
		BindOutput(tensorInfo.Output1Name, binding, output1TensorOwner.Tensor, output1Info.Value.Dimensions64);
		return new RawOutput(output0TensorOwner, output1TensorOwner);
	}

	public DenseTensor<float> Output0 => _output0.Tensor;
	public DenseTensor<float>? Output1 => _output1?.Tensor;

	public void Dispose()
	{
		_output0.Dispose();
		_output1?.Dispose();
	}
	
	internal int Version { get; set; }

	internal RawOutput(DenseTensorOwner<float> output0, DenseTensorOwner<float>? output1 = null)
	{
		_output0 = output0;
		_output1 = output1;
	}

	private readonly DenseTensorOwner<float> _output0;
	private readonly DenseTensorOwner<float>? _output1;

	private static void BindOutput(string name, OrtIoBinding binding, DenseTensor<float> tensor, long[] shape)
	{
		var value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, shape);
		binding.BindOutput(name, value);
	}
}