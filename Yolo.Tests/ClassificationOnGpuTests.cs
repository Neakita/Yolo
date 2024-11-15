using Xunit.Abstractions;
using Yolo.Tests.Data;
using Yolo.Tests.Helpers;

namespace Yolo.Tests;

[Collection("gpu")]
public sealed class ClassificationOnGpuTests
{
	public static IEnumerable<string> Models => ClassificationTestsData.Models;
	public static IEnumerable<ClassificationTestData> TestData => ClassificationTestsData.TestData;

	public ClassificationOnGpuTests(ITestOutputHelper testOutputHelper)
	{
		_testHelper = new ClassificationTestHelper(testOutputHelper, false);
	}

	[Theory, CombinatorialData]
	public void ShouldClassifyOnGpuWhenSizeMatches(
		[CombinatorialMemberData(nameof(Models))] string modelFileName,
		[CombinatorialMemberData(nameof(TestData))] ClassificationTestData testData)
	{
		_testHelper.PredictPlotAndAssert(modelFileName, testData);
	}

	private readonly ClassificationTestHelper _testHelper;
}