using Xunit.Abstractions;
using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

public sealed class PoseTests
{
	public static IEnumerable<ImageDetectionExpectation> MatchingSizesExpectations =>
		PoseTestsData.MatchingSizesExpectations;

	public static IEnumerable<ImageDetectionExpectation> HigherInputSizesExpectations =>
		PoseTestsData.HigherInputSizesExpectations;

	public PoseTests(ITestOutputHelper testOutputHelper)
	{
		_poseTestHelper = new PoseTestHelper(testOutputHelper, false);
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