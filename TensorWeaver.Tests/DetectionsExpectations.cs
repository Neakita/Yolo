using TensorWeaver.Tests.Inference.Detectors;

namespace TensorWeaver.Tests;

public static class DetectionsExpectations
{
	public static readonly IReadOnlyCollection<DetectedObjectExpectation> BusCOCOExpectations =
	[
		new("person", 2, 4), // 2 persons are clearly visible and 2 more are only partially
		new("bus", 1), // bus is huge object on the image and locates in the center
		new("stop sign", 0, 1) // the sign is barely noticeable
	];
}