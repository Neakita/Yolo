using Xunit.Abstractions;
using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class PoseOnGpuTests
{
	public static IEnumerable<ImageDetectionExpectation> MatchingSizesExpectations =>
		PoseTestsData.MatchingSizesExpectations;

	public static IEnumerable<ImageDetectionExpectation> HigherInputSizesExpectations =>
		PoseTestsData.HigherInputSizesExpectations;

	public PoseOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, true);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenSizeMatches([CombinatorialMemberData(nameof(MatchingSizesExpectations))] ImageDetectionExpectation expectation)
	{
		_poseTestHelper.PredictPlotAndAssert(expectation);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenImageSizeIsHigher([CombinatorialMemberData(nameof(HigherInputSizesExpectations))] ImageDetectionExpectation expectation)
	{
		_poseTestHelper.PredictPlotAndAssert(expectation);
	}

	private readonly PoseTestHelper _poseTestHelper;
}