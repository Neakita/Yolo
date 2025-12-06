using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
		_ioBinding = Session.CreateIoBinding();
		_output = new RawOutput(Session, _ioBinding);
		var inputShape = TensorInfo.GetInputInfo(Session).Shape;
		_inputTensor = inputShape.AllocateTensor();
		_inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			_inputTensor.Buffer,
			inputShape.Dimensions64);
		_ioBinding.BindInput(Session.InputNames.Single(), _inputValue);
	}

	public void SetInput<TPixel>(
		ReadOnlySpan2D<TPixel> data,
		InputProcessor<TPixel> processor)
		where TPixel : unmanaged
	{
		processor.ProcessInput(data, _inputTensor);
		_ioBinding.BindInput(Session.InputNames.Single(), _inputValue);
	}

	public void Predict()
	{
		Session.RunWithBinding(_runOptions, _ioBinding);
	}

	public TResult GetOutput<TResult>(OutputProcessor<TResult> processor)
	{
		return processor.Process(_output);
	}

	public void Dispose()
	{
		_inputValue.Dispose();
		_runOptions.Dispose();
		_ioBinding.Dispose();
		Session.Dispose();
	}

	private readonly OrtValue _inputValue;
	private readonly RunOptions _runOptions = new();
	private readonly RawOutput _output;
	private readonly DenseTensor<float> _inputTensor;
	private readonly OrtIoBinding _ioBinding;
}