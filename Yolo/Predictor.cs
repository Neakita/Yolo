using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Yolo;

public sealed class Predictor : IDisposable
{
	public RawOutput Output { get; }
	public Metadata Metadata { get; }

	public Predictor(byte[] model, SessionOptions sessionOptions)
	{
		_session = new InferenceSession(model, sessionOptions);
		Metadata = new Metadata(_session);
		if (Metadata.Task == Task.Pose)
			PoserMetadata = new PoserMetadata(_session);
		_tensorInfo = new TensorInfo(_session);
		Guard.IsEqualTo(_tensorInfo.Input.Dimensions[0], Metadata.BatchSize);
		_ioBinding = _session.CreateIoBinding();
		_inputTensorOwner = DenseTensorOwner<float>.Allocate(_tensorInfo.Input);
		BindInput(_inputTensorOwner.Tensor, _ioBinding);
		Output = RawOutput.Create(_ioBinding, _tensorInfo);
	}

	public void Predict<TPixel>(
		ReadOnlySpan<TPixel> data,
		InputProcessor<TPixel> inputProcessor)
		where TPixel : unmanaged
	{
		ValidateDataLength(data.Length);
		inputProcessor.ProcessInput(data, _inputTensorOwner.Tensor);
		_session.RunWithBinding(_runOptions, _ioBinding);
		Output.Version++;
	}

	public void Dispose()
	{
		_inputTensorOwner.Dispose();
		Output.Dispose();
		_ioBinding.Dispose();
		_session.Dispose();
	}
	
	internal PoserMetadata? PoserMetadata { get; }

	private readonly InferenceSession _session;
	private readonly TensorInfo _tensorInfo;
	private readonly RunOptions _runOptions = new();
	private readonly OrtIoBinding _ioBinding;
	private readonly DenseTensorOwner<float> _inputTensorOwner;

	private void BindInput(DenseTensor<float> tensor, OrtIoBinding binding)
	{
		var inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			tensor.Buffer,
			_tensorInfo.Input.Dimensions64);
		binding.BindInput(_session.InputNames.Single(), inputValue);
	}

	private void ValidateDataLength(int length)
	{
		Guard.IsEqualTo(length, Metadata.ImageSize.Width * Metadata.ImageSize.Height * Metadata.BatchSize);
	}
}