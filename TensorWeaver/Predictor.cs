using System.Buffers;
using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using TensorWeaver.InputData;
using TensorWeaver.InputProcessing;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;
using TensorWeaver.OutputProcessing;

namespace TensorWeaver;

public sealed class Predictor : IDisposable
{
	public InferenceSession Session { get; }

	public Predictor(byte[] modelData, SessionOptions sessionOptions)
	{
		Session = new InferenceSession(modelData, sessionOptions);
		var tensorInfo = new TensorInfo(Session);
		_ioBinding = Session.CreateIoBinding();
		_output = RawOutput.Create(_ioBinding, tensorInfo);
		_inputTensorOwner = DenseTensorOwner<float>.Allocate(tensorInfo.Input);
		_inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			_inputTensorOwner.Tensor.Buffer,
			tensorInfo.Input.Dimensions64);
		var dimensions = Session.InputMetadata.Values.Single().Dimensions;
		_imageSize = new Vector2D<int>(dimensions[3], dimensions[2]);
	}

	public TResult Predict<TPixel, TResult>(
		ReadOnlySpan2D<TPixel> data,
		InputProcessor<TPixel> inputProcessor,
		OutputProcessor<TResult> outputProcessor)
		where TPixel : unmanaged
	{
		ProcessInput(data, inputProcessor);
		_ioBinding.BindInput(Session.InputNames.Single(), _inputValue);
		Session.RunWithBinding(_runOptions, _ioBinding);
		return outputProcessor.Process(_output);
	}

	public void Dispose()
	{
		_inputValue.Dispose();
		_runOptions.Dispose();
		_output.Dispose();
		_inputTensorOwner.Dispose();
		_ioBinding.Dispose();
		Session.Dispose();
	}

	private readonly OrtValue _inputValue;
	private readonly RunOptions _runOptions = new();
	private readonly RawOutput _output;
	private readonly DenseTensorOwner<float> _inputTensorOwner;
	private readonly OrtIoBinding _ioBinding;
	private readonly Vector2D<int> _imageSize;

	private void ProcessInput<TPixel>(ReadOnlySpan2D<TPixel> data, InputProcessor<TPixel> inputProcessor) where TPixel : unmanaged
	{
		if (data.Width == _imageSize.X && data.Height == _imageSize.Y)
			inputProcessor.ProcessInput(data, _inputTensorOwner.Tensor);
		else
		{
			var bufferArray = ArrayPool<TPixel>.Shared.Rent(_imageSize.X * _imageSize.Y);
			Span2D<TPixel> bufferSpan = new(bufferArray, _imageSize.Y, _imageSize.X);
			NearestNeighbourImageResizer.Resize(data, bufferSpan);
			inputProcessor.ProcessInput(bufferSpan, _inputTensorOwner.Tensor);
			ArrayPool<TPixel>.Shared.Return(bufferArray);
		}
	}
}