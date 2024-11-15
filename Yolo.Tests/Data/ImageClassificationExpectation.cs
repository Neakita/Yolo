namespace Yolo.Tests.Data;

public sealed class ImageClassificationExpectation
{
	public string ImageFileName { get; }
	public string ExpectedClassification { get; }

	public ImageClassificationExpectation(string imageFileName, string expectedClassification)
	{
		ImageFileName = imageFileName;
		ExpectedClassification = expectedClassification;
	}
}