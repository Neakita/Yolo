namespace TensorWeaver.Tests.Inference;

public interface PredictorOutputHandler
{
	Task HandleOutputAsync(Predictor predictor, CancellationToken cancellationToken);
}