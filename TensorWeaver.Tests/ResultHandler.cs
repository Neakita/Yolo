namespace TensorWeaver.Tests;

public interface ResultHandler<in TResult>
{
	Task HandleResultAsync(TResult result, CancellationToken cancellationToken);
}