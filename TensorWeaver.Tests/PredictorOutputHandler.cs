namespace TensorWeaver.Tests;

public interface PredictorOutputHandler
{
	Task HandleOutputAsync(Predictor predictor, CancellationToken cancellationToken);
}