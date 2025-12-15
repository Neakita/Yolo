using FluentAssertions;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Inference.Classificators;

public sealed class ClassificationAsserter : ResultHandler<IEnumerable<Classification>>
{
	public ClassificationAsserter(IReadOnlyList<string> classesNames, string expectedClassName)
	{
		_classesNames = classesNames;
		_expectedClassName = expectedClassName;
	}

	public Task HandleResultAsync(IEnumerable<Classification> result, CancellationToken cancellationToken)
	{
		var classification = result.First();
		var className = _classesNames[classification.ClassId];
		className.Should().Be(_expectedClassName);
		return Task.CompletedTask;
	}

	private readonly IReadOnlyList<string> _classesNames;
	private readonly string _expectedClassName;
}