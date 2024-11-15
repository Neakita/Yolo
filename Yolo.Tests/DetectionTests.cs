using Xunit.Abstractions;
using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

public class DetectionTests
{
	public static IEnumerable<ImageDetectionExpectation> MatchingSizesExpectations =>
		DetectionTestsData.MatchingSizesExpectations;

	public static IEnumerable<ImageDetectionExpectation> HigherInputSizesExpectations =>
		DetectionTestsData.HigherInputSizesExpectations;

	public DetectionTests(ITestOutputHelper testOutputHelper)
	{
		_detectionTestHelper = new DetectionTestHelper(testOutputHelper, false);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenSizeMatches([CombinatorialMemberData(nameof(MatchingSizesExpectations))] ImageDetectionExpectation expectation)
	{
		_detectionTestHelper.PredictPlotAndAssert(expectation);
	}

	[Theory, CombinatorialData]
	public void ShouldDetectOnGpuWhenImageSizeIsHigher([CombinatorialMemberData(nameof(HigherInputSizesExpectations))] ImageDetectionExpectation expectation)
	{
		_detectionTestHelper.PredictPlotAndAssert(expectation);
	}

	private readonly DetectionTestHelper _detectionTestHelper;
}