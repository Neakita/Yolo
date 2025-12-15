namespace TensorWeaver.Tests.Inference.Detectors;

public sealed class DetectedObjectExpectation
{
	public string ClassName { get; }
	public byte MinimumCount { get; }
	public byte MaximumCount { get; }

	public DetectedObjectExpectation(string className, byte count)
	{
		ClassName = className;
		MinimumCount = count;
		MaximumCount = count;
	}

	public DetectedObjectExpectation(string className, byte minimumCount, byte maximumCount)
	{
		ClassName = className;
		MinimumCount = minimumCount;
		MaximumCount = maximumCount;
	}
}