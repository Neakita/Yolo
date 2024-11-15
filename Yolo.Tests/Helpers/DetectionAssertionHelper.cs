using FluentAssertions;
using Yolo.Metadata;
using Yolo.OutputData;
using Yolo.Tests.Data;

namespace Yolo.Tests.Helpers;

internal static class DetectionAssertionHelper
{
	public static void AssertClassifications(
		ModelMetadata metadata,
		IReadOnlyCollection<DetectedObjectExpectation> expectations,
		IReadOnlyCollection<Classification> actualDetections)
	{
		foreach (var expectation in expectations)
			expectation.Assert(actualDetections, metadata);
		actualDetections.Should().OnlyContain(classification => expectations.Any(expectation => expectation.ClassName == metadata.ClassesNames[classification.ClassId]));
	}
}