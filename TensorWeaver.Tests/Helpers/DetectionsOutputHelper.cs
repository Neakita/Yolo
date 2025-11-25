using TensorWeaver.Metadata;
using TensorWeaver.OutputData;

namespace TensorWeaver.Tests.Helpers;

internal static class DetectionsOutputHelper
{
	public static void WriteClassifications(ITestOutputHelper outputHelper, ModelMetadata metadata, IEnumerable<Classification> classifications)
	{
		foreach (var classification in classifications)
			outputHelper.WriteLine($"{metadata.ClassesNames[classification.ClassId]}: {classification.Confidence:P1}");
	}
}