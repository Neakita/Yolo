using FluentAssertions;
using TensorWeaver.Metadata;
using TensorWeaver.OutputData;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests.Data;

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

	public void Assert(IEnumerable<Classification> detections, YoloMetadata metadata)
	{
		var detectionsCount = detections.Count(detection => metadata.ClassesNames[detection.ClassId] == ClassName);
		detectionsCount.Should().BeInRange(MinimumCount, MaximumCount);
	}
}