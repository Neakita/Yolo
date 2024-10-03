namespace Yolo.Tests;

internal sealed class PredictionExpectation
{
	public string ClassName { get; }
	public int MinimumCount { get; }
	public int MaximumCount { get; }

	public PredictionExpectation(string className, int minimumCount, int maximumCount)
	{
		ClassName = className;
		MinimumCount = minimumCount;
		MaximumCount = maximumCount;
	}
}