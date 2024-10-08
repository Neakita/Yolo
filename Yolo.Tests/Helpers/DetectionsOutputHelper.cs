using Xunit.Abstractions;

namespace Yolo.Tests.Helpers;

internal static class DetectionsOutputHelper
{
	public static void WriteClassifications(ITestOutputHelper outputHelper, Metadata metadata, IEnumerable<Classification> classifications)
	{
		foreach (var classification in classifications)
			outputHelper.WriteLine($"{metadata.ClassesNames[classification.ClassId]}: {classification.Confidence:P1}");
	}
}