using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TensorWeaver.Metadata;

namespace TensorWeaver.OutputData;

public sealed class RawOutput
{
	internal static RawOutput Create(OrtIoBinding binding, TensorInfo tensorInfo)
	{
		var output0Info = tensorInfo.Output0;
		var output1Info = tensorInfo.Output1;
		var output0Tensor = output0Info.AllocateTensor();
		BindOutput(tensorInfo.Output0Name, binding, output0Tensor, output0Info.Dimensions64);
		if (output1Info == null)
			return new RawOutput(output0Tensor);
		Guard.IsNotNull(tensorInfo.Output1Name);
		var output1Tensor = output1Info.Value.AllocateTensor();
		BindOutput(tensorInfo.Output1Name, binding, output1Tensor, output1Info.Value.Dimensions64);
		return new RawOutput(output0Tensor, output1Tensor);
	}

	public DenseTensor<float> Output0 { get; }
	public DenseTensor<float>? Output1 { get; }

	private RawOutput(DenseTensor<float> output0, DenseTensor<float>? output1 = null)
	{
		Output0 = output0;
		Output1 = output1;
	}

	private static void BindOutput(string name, OrtIoBinding binding, DenseTensor<float> tensor, long[] shape)
	{
		var value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, shape);
		binding.BindOutput(name, value);
	}
}