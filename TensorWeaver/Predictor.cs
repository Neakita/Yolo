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
		_ioBinding.BindInput(Session.InputNames.Single(), _inputValue);
	}

	public void SetInput<TPixel>(
		ReadOnlySpan2D<TPixel> data,
		InputProcessor<TPixel> inputProcessor)
		where TPixel : unmanaged
	{
		inputProcessor.ProcessInput(data, _inputTensorOwner.Tensor);
		_ioBinding.BindInput(Session.InputNames.Single(), _inputValue);
	}

	public TResult Predict<TResult>(OutputProcessor<TResult> outputProcessor)
	{
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
}