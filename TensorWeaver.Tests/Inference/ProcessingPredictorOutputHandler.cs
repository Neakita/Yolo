using TensorWeaver.OutputProcessing;

namespace TensorWeaver.Tests.Inference;

public sealed class ProcessingPredictorOutputHandler<TResult> : PredictorOutputHandler
{
	public ProcessingPredictorOutputHandler(OutputProcessor<TResult> outputProcessor, ResultHandler<TResult> resultHandler)
	{
		_outputProcessor = outputProcessor;
		_resultHandler = resultHandler;
	}

	public Task HandleOutputAsync(Predictor predictor, CancellationToken cancellationToken)
	{
		var result = _outputProcessor.Process(predictor.Output);
		return _resultHandler.HandleResultAsync(result, cancellationToken);
	}

	private readonly OutputProcessor<TResult> _outputProcessor;
	private readonly ResultHandler<TResult> _resultHandler;
}