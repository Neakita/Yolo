using FluentAssertions;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;
using TensorWeaver.Tests.Data;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Helpers;

internal static class DetectionAssertionHelper
{
	public static void AssertClassifications(
		YoloMetadata metadata,
		IReadOnlyCollection<DetectedObjectExpectation> expectations,
		IReadOnlyCollection<Classification> detections)
	{
		foreach (var expectation in expectations)
			expectation.Assert(detections, metadata);
		AssertDoesNotContainExcessiveObjects(metadata, expectations, detections);
	}

	private static void AssertDoesNotContainExcessiveObjects(
		YoloMetadata metadata,
		IReadOnlyCollection<DetectedObjectExpectation> expectations,
		IReadOnlyCollection<Classification> detections)
	{
		detections.Should().OnlyContain(classification => expectations.Any(expectation => expectation.ClassName == metadata.ClassesNames[classification.ClassId]));
	}
}