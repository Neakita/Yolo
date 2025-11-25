namespace TensorWeaver.Tests.Data;

public sealed class ClassificationTestData
{
	public string ImageFileName { get; }
	public string ExpectedClassification { get; }

	public ClassificationTestData(string imageFileName, string expectedClassification)
	{
		ImageFileName = imageFileName;
		ExpectedClassification = expectedClassification;
	}
}