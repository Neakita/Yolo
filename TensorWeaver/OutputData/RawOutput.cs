using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TensorWeaver.Metadata;

namespace TensorWeaver.OutputData;

public sealed class RawOutput : IDisposable
{
	public IReadOnlyList<DenseTensor<float>> Tensors { get; }

	public void Dispose()
	{
		foreach (var disposable in _disposables)
			disposable.Dispose();
	}

	internal RawOutput(InferenceSession session, OrtIoBinding binding)
	{
		var tensorsInfo = TensorInfo.GetOutputInfo(session);
		var tensors = new List<DenseTensor<float>>(tensorsInfo.Count);
		_disposables = new List<IDisposable>(tensorsInfo.Count);
		foreach (var tensorInfo in tensorsInfo)
		{
			var tensor = tensorInfo.Shape.AllocateTensor();
			tensors.Add(tensor);
			var disposable = BindOutput(tensorInfo.Name, binding, tensor, tensorInfo.Shape.Dimensions64);
			_disposables.Add(disposable);
		}
		Tensors = tensors;
	}

	private readonly List<IDisposable> _disposables;

	private static IDisposable BindOutput(string name, OrtIoBinding binding, DenseTensor<float> tensor, long[] shape)
	{
		var value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, shape);
		binding.BindOutput(name, value);
		return value;
	}
}