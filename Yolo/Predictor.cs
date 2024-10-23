using System.Buffers;
using CommunityToolkit.Diagnostics;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using Yolo.InputData;
using Yolo.InputProcessing;
using Yolo.Metadata;
using Yolo.OutputData;
using Yolo.OutputProcessing;
using ModelMetadata = Yolo.Metadata.ModelMetadata;
using Task = Yolo.Metadata.Task;

namespace Yolo;

public sealed class Predictor : IDisposable
{
	public ModelMetadata Metadata { get; }

	public Predictor(byte[] modelData, SessionOptions sessionOptions)
	{
		_session = new InferenceSession(modelData, sessionOptions);
		Metadata = new ModelMetadata(_session);
		if (Metadata.Task == Task.Pose)
			PoserMetadata = new PoserMetadata(_session);
		var tensorInfo = new TensorInfo(_session);
		Guard.IsEqualTo(tensorInfo.Input.Dimensions[0], Metadata.BatchSize);
		_ioBinding = _session.CreateIoBinding();
		_output = RawOutput.Create(_ioBinding, tensorInfo);
		_inputTensorOwner = DenseTensorOwner<float>.Allocate(tensorInfo.Input);
		_inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			_inputTensorOwner.Tensor.Buffer,
			tensorInfo.Input.Dimensions64);
	}

	public IReadOnlyList<TResult> Predict<TPixel, TResult>(
		ReadOnlySpan2D<TPixel> data,
		InputProcessor<TPixel> inputProcessor,
		OutputProcessor<TResult> outputProcessor)
		where TPixel : unmanaged
	{
		ProcessInput(data, inputProcessor);
		_ioBinding.BindInput(_session.InputNames.Single(), _inputValue);
		_session.RunWithBinding(_runOptions, _ioBinding);
		return outputProcessor.Process(_output);
	}

	public void Dispose()
	{
		_inputValue.Dispose();
		_runOptions.Dispose();
		_output.Dispose();
		_inputTensorOwner.Dispose();
		_ioBinding.Dispose();
		_session.Dispose();
	}

	internal PoserMetadata? PoserMetadata { get; }

	private readonly OrtValue _inputValue;
	private readonly RunOptions _runOptions = new();
	private readonly RawOutput _output;
	private readonly DenseTensorOwner<float> _inputTensorOwner;
	private readonly OrtIoBinding _ioBinding;
	private readonly InferenceSession _session;

	private void ProcessInput<TPixel>(ReadOnlySpan2D<TPixel> data, InputProcessor<TPixel> inputProcessor) where TPixel : unmanaged
	{
		if (data.Width == Metadata.ImageSize.X && data.Height == Metadata.ImageSize.Y)
			inputProcessor.ProcessInput(data, _inputTensorOwner.Tensor);
		else
		{
			var bufferArray = ArrayPool<TPixel>.Shared.Rent(Metadata.ImageSize.X * Metadata.ImageSize.Y);
			Span2D<TPixel> bufferSpan = new(bufferArray, Metadata.ImageSize.Y, Metadata.ImageSize.X);
			NearestNeighbourImageResizer.Resize(data, bufferSpan);
			inputProcessor.ProcessInput(bufferSpan, _inputTensorOwner.Tensor);
			ArrayPool<TPixel>.Shared.Return(bufferArray);
		}
	}
}