using TensorWeaver.RFDETR;
using TensorWeaver.Yolo;

namespace TensorWeaver.Tests;

internal static class TestCases
{
	public static IEnumerable<TestCase> All =>
	[
		..RFDETRNano,
		..YoloV8Nano,
		..YoloV8NanoUInt8,
		..YoloV10Nano,
		..YoloV11Nano
	];

	private static IEnumerable<TestCase> RFDETRNano =>
		TestCase.Create(
			Models.RFDETRNano,
			BusImageInfo,
			new RFDETRDetectionProcessor(),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV8Nano =>
		TestCase.Create(
			Models.YoloV8Nano,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)Models.YoloV8Nano.ClassesNames.Count, new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV8NanoUInt8 =>
		TestCase.Create(
			Models.YoloV8NanoUInt8,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)Models.YoloV8NanoUInt8.ClassesNames.Count, new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV10Nano =>
		TestCase.Create(
			Models.YoloV10Nano,
			BusImageInfo,
			new YoloV10DetectionsProcessor(new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static IEnumerable<TestCase> YoloV11Nano =>
		TestCase.Create(
			Models.YoloV11Nano,
			BusImageInfo,
			new YoloV8DetectionsProcessor((byte)Models.YoloV8NanoUInt8.ClassesNames.Count, new Vector2D<int>(800, 800)),
			BusImageObjectExpectations);

	private static readonly IReadOnlyCollection<DetectedObjectExpectation> BusImageObjectExpectations =
	[
		new("person", 2, 4), // 2 persons are clearly visible and 2 more are only partially
		new("bus", 1), // bus is huge object on the image and locates in the center
		new("stop sign", 0, 1) // the sign is barely noticeable
	];

	private static readonly ImageInfo BusImageInfo = new("bus.png");
}