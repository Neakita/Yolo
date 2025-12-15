using TensorWeaver.Tests.Inference.Detectors;

namespace TensorWeaver.Tests.Inference;

internal static class TestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..DetectorTestCases.All
	];
}