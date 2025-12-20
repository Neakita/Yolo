using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TensorWeaver.InputProcessing;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;

namespace TensorWeaver;

public sealed class Predictor : IDisposable
{
	public InferenceSession Session { get; }
	public RawOutput Output { get; }

	public Predictor(byte[] modelData, SessionOptions sessionOptions)
	{
		Session = new InferenceSession(modelData, sessionOptions);
		_ioBinding = Session.CreateIoBinding();
		Output = new RawOutput(Session, _ioBinding);
		var inputShape = TensorInfo.GetInputInfo(Session).Shape;
		_inputTensor = inputShape.AllocateTensor();
		_inputValue = OrtValue.CreateTensorValueFromMemory(
			OrtMemoryInfo.DefaultInstance,
			_inputTensor.Buffer,
			inputShape.Dimensions64);
		_inputName = Session.InputNames.Single();
	}

	public void SetInput<TPixel>(
		ReadOnlySpan2D<TPixel> data,
		InputProcessor<TPixel> processor)
		where TPixel : unmanaged
	{
		processor.ProcessInput(data, _inputTensor);
		_ioBinding.BindInput(_inputName, _inputValue);
	}

	public void Predict()
	{
		Session.RunWithBinding(_runOptions, _ioBinding);
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
	private readonly DenseTensor<float> _inputTensor;
	private readonly OrtIoBinding _ioBinding;
	private readonly string _inputName;
}