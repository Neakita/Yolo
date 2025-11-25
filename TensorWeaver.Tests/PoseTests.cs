using TensorWeaver.Tests.Data;
using TensorWeaver.Tests.Helpers;

namespace TensorWeaver.Tests;

public sealed class PoseTests
{
	public static IEnumerable<DetectionTestData> MatchingSizesExpectations =>
		PoseTestsData.MatchingSizesExpectations;

	public static IEnumerable<DetectionTestData> HigherInputSizesExpectations =>
		PoseTestsData.HigherInputSizesExpectations;

	public PoseTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, false);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenSizeMatches([CombinatorialMemberData(nameof(MatchingSizesExpectations))] DetectionTestData testData)
	{
		_poseTestHelper.PredictPlotAndAssert(testData);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenImageSizeIsHigher([CombinatorialMemberData(nameof(HigherInputSizesExpectations))] DetectionTestData testData)
	{
		_poseTestHelper.PredictPlotAndAssert(testData);
	}

	private readonly PoseTestHelper _poseTestHelper;
}