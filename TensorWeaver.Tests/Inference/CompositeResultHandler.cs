namespace TensorWeaver.Tests.Inference;

public sealed class CompositeResultHandler<TResult> : ResultHandler<TResult>
{
	public CompositeResultHandler(params IReadOnlyCollection<ResultHandler<TResult>> handlers)
	{
		_handlers = handlers;
	}

	public async Task HandleResultAsync(TResult result, CancellationToken cancellationToken)
	{
		foreach (var handler in _handlers)
			await handler.HandleResultAsync(result, cancellationToken);
	}

	private readonly IReadOnlyCollection<ResultHandler<TResult>> _handlers;
}