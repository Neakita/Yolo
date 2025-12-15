using FluentAssertions;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests;

public sealed class DetectionsAsserter : ResultHandler<IReadOnlyCollection<Detection>>
{
	public DetectionsAsserter(IReadOnlyCollection<DetectedObjectExpectation> expectations, IReadOnlyList<string> classesNames)
	{
		_expectations = expectations;
		_classesNames = classesNames;
	}

	public Task HandleResultAsync(IReadOnlyCollection<Detection> detections, CancellationToken cancellationToken)
	{
		foreach (var expectation in _expectations)
		{
			var matchedDetectionsCount = detections.Count(detection => _classesNames[detection.ClassId] == expectation.ClassName);
			matchedDetectionsCount.Should().BeInRange(
				expectation.MinimumCount,
				expectation.MaximumCount,
				"expected that much detections for class \'{0}\'",
				expectation.ClassName);
		}
		return Task.CompletedTask;
	}

	private readonly IReadOnlyCollection<DetectedObjectExpectation> _expectations;
	private readonly IReadOnlyList<string> _classesNames;
}