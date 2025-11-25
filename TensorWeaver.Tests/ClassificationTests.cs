using TensorWeaver.Tests.Data;
using TensorWeaver.Tests.Helpers;

namespace TensorWeaver.Tests;

public sealed class ClassificationTests
{
	public static IEnumerable<string> Models => ClassificationTestsData.Models;
	public static IEnumerable<ClassificationTestData> Expectations => ClassificationTestsData.TestData;

	public ClassificationTests(ITestOutputHelper testOutputHelper)
	{
		_testHelper = new ClassificationTestHelper(testOutputHelper, false);
	}

	[Theory, CombinatorialData]
	public void ShouldClassifyWhenSizeMatches(
		[CombinatorialMemberData(nameof(Models))] string modelFileName,
		[CombinatorialMemberData(nameof(Expectations))] ClassificationTestData data)
	{
		_testHelper.PredictPlotAndAssert(modelFileName, data);
	}

	private readonly ClassificationTestHelper _testHelper;
}