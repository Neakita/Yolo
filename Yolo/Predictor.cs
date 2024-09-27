using System.Collections.Immutable;
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
		_tensorInfo = new TensorInfo(_session);
		Guard.IsEqualTo(_tensorInfo.Input.Dimensions[0], Metadata.BatchSize);
	}

	public ImmutableArray<TPrediction> Predict<TPixel, TPrediction>(
		ReadOnlySpan<TPixel> data,
		InputProcessor<TPixel> inputProcessor,
		OutputProcessor<TPrediction> outputProcessor)
		where TPixel : unmanaged
	{
		ValidateDataLength(data.Length);
		using var binding = _session.CreateIoBinding();
		using var inputTensorOwner = DenseTensorOwner<float>.Allocate(_tensorInfo.Input);
		Guard.IsNotNull(inputProcessor);
		inputProcessor.ProcessInput(data, inputTensorOwner.Tensor);
		BindInput(inputTensorOwner.Tensor, binding);
		using var output = RawOutput.Create(binding, _tensorInfo);
		_session.RunWithBinding(_runOptions, binding);
		return outputProcessor.Process(output);
	}

	private void BindInput(DenseTensor<float> tensor, OrtIoBinding binding)
	{
		var inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			tensor.Buffer,
			_tensorInfo.Input.Dimensions64);
		binding.BindInput(_session.InputNames.Single(), inputValue);
	}

	public void Dispose()
	{
		_session.Dispose();
	}

	private readonly InferenceSession _session;
	private readonly TensorInfo _tensorInfo;
	private readonly RunOptions _runOptions = new();

	private void ValidateDataLength(int length)
	{
		Guard.IsEqualTo(length, Metadata.ImageSize.Width * Metadata.ImageSize.Height * Metadata.BatchSize);
	}
}