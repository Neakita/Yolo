using TensorWeaver.Tests.Inference.Classificators;
using TensorWeaver.Tests.Inference.Detectors;
using TensorWeaver.Tests.Inference.Posers;

namespace TensorWeaver.Tests.Inference;

internal static class TestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..DetectorTestCases.All,
		..ClassificatorTestCases.All,
		..PoseTestCases.All
	];
}