using Xunit.Abstractions;
using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class DetectionOnGpuTests
{
	public static IEnumerable<ImageDetectionExpectation> MatchingSizesExpectations =>
		DetectionTestsData.MatchingSizesExpectations;

	public static IEnumerable<ImageDetectionExpectation> HigherInputSizesExpectations =>
		DetectionTestsData.HigherInputSizesExpectations;

	public DetectionOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_detectionTestHelper = new DetectionTestHelper(testOutputHelper, true);
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