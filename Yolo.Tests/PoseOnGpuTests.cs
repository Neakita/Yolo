using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class PoseOnGpuTests
{
	public static IEnumerable<DetectionTestData> MatchingSizesExpectations =>
		PoseTestsData.MatchingSizesExpectations;

	public static IEnumerable<DetectionTestData> HigherInputSizesExpectations =>
		PoseTestsData.HigherInputSizesExpectations;

	public PoseOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, true);
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