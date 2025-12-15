namespace TensorWeaver.Tests.Inference;

public interface ResultHandler<in TResult>
{
	Task HandleResultAsync(TResult result, CancellationToken cancellationToken);
}