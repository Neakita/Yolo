using CommunityToolkit.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Yolo.OutputProcessing;

namespace Yolo;

public sealed class Predictor : IDisposable
{
	public Metadata Metadata { get; }

	public Predictor(byte[] model, SessionOptions sessionOptions)
	{
		_session = new InferenceSession(model, sessionOptions);
		Metadata = new Metadata(_session);
		if (Metadata.Task == Task.Pose)
			PoserMetadata = new PoserMetadata(_session);
		_tensorInfo = new TensorInfo(_session);
		Guard.IsEqualTo(_tensorInfo.Input.Dimensions[0], Metadata.BatchSize);
	}

	public IReadOnlyList<TResult> Predict<TPixel, TResult>(
		ReadOnlySpan<TPixel> data,
		InputProcessor<TPixel> inputProcessor,
		OutputProcessor<TResult> outputProcessor)
		where TPixel : unmanaged
	{
		ValidateDataLength(data.Length);
		using var ioBinding = _session.CreateIoBinding();
		using var output = RawOutput.Create(ioBinding, _tensorInfo);
		using var inputTensorOwner = DenseTensorOwner<float>.Allocate(_tensorInfo.Input);
		inputProcessor.ProcessInput(data, inputTensorOwner.Tensor);
		BindInput(inputTensorOwner.Tensor, ioBinding);
		_session.RunWithBinding(_runOptions, ioBinding);
		return outputProcessor.Process(output);
	}

	public void Dispose()
	{
		_session.Dispose();
	}

	internal PoserMetadata? PoserMetadata { get; }

	private readonly InferenceSession _session;
	private readonly TensorInfo _tensorInfo;
	private readonly RunOptions _runOptions = new();

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