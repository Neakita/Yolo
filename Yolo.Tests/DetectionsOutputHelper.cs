using Xunit.Abstractions;

namespace Yolo.Tests;

internal static class DetectionsOutputHelper
{
	public static void WriteDetections(ITestOutputHelper outputHelper, Metadata metadata, IReadOnlyCollection<Detection> detections)
	{
		foreach (var detection in detections)
			outputHelper.WriteLine($"{metadata.ClassesNames[detection.ClassId]}: {detection.Confidence:P1}");
	}
}