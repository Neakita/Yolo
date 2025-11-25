using TensorWeaver.Tests.Data;
using TensorWeaver.Tests.Helpers;

namespace TensorWeaver.Tests;

[Collection("gpu")]
public sealed class DetectionOnGpuTests
{
	public static IEnumerable<DetectionTestData> MatchingSizesExpectations =>
		DetectionTestsData.MatchingSizesExpectations;

	public static IEnumerable<DetectionTestData> HigherInputSizesExpectations =>
		DetectionTestsData.HigherInputSizesExpectations;

	public DetectionOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_detectionTestHelper = new DetectionTestHelper(testOutputHelper, true);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenSizeMatches([CombinatorialMemberData(nameof(MatchingSizesExpectations))] DetectionTestData testData)
	{
		_detectionTestHelper.PredictPlotAndAssert(testData);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenImageSizeIsHigher([CombinatorialMemberData(nameof(HigherInputSizesExpectations))] DetectionTestData testData)
	{
		_detectionTestHelper.PredictPlotAndAssert(testData);
	}

	private readonly DetectionTestHelper _detectionTestHelper;
}