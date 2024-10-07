using Xunit.Abstractions;

namespace Yolo.Tests.Helpers;

internal static class DetectionsOutputHelper
{
	public static void WriteDetections(ITestOutputHelper outputHelper, Metadata metadata, IReadOnlyCollection<Detection> detections)
	{
		foreach (var detection in detections)
			outputHelper.WriteLine($"{metadata.ClassesNames[detection.ClassId]}: {detection.Confidence:P1}");
	}

	public static void WriteClassifications(ITestOutputHelper outputHelper, Metadata metadata, IReadOnlyCollection<Classification> classifications)
	{
		foreach (var classification in classifications)
			outputHelper.WriteLine($"{metadata.ClassesNames[classification.ClassId]}: {classification.Confidence:P1}");
	}
}