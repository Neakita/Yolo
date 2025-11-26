namespace TensorWeaver.Tests.Data;

public static class DetectionTestsData
{
	public static IReadOnlyCollection<ModelInfo> Models =>
	[
		new()
		{
			Name = "yolov8n-uint8.onnx",
			Resolution = 640
		},
		new()
		{
			Name = "yolov8n160fp32.onnx",
			Resolution = 160
		},
		new()
		{
			Name = "yolov8n224fp32.onnx",
			Resolution = 224
		},
		new()
		{
			Name = "yolov8n320fp32.onnx",
			Resolution = 320
		},
		new()
		{
			Name = "yolov8n480fp32.onnx",
			Resolution = 480
		},
		new()
		{
			Name = "yolov8n640fp32.onnx",
			Resolution = 640
		},
		new()
		{
			Name = "yolov8n800fp32.onnx",
			Resolution = 800
		},
		new()
		{
			Name = "yolov10n160fp32.onnx",
			Resolution = 160
		},
		new()
		{
			Name = "yolov10n224fp32.onnx",
			Resolution = 224
		},
		new()
		{
			Name = "yolov10n320fp32.onnx",
			Resolution = 320
		},
		new()
		{
			Name = "yolov10n480fp32.onnx",
			Resolution = 480
		},
		new()
		{
			Name = "yolov10n640fp32.onnx",
			Resolution = 640
		},
		new()
		{
			Name = "yolov10n800fp32.onnx",
			Resolution = 800
		},
		new()
		{
			Name = "yolo11n160fp32.onnx",
			Resolution = 160
		},
		new()
		{
			Name = "yolo11n224fp32.onnx",
			Resolution = 224
		},
		new()
		{
			Name = "yolo11n320fp32.onnx",
			Resolution = 320
		},
		new()
		{
			Name = "yolo11n480fp32.onnx",
			Resolution = 480
		},
		new()
		{
			Name = "yolo11n640fp32.onnx",
			Resolution = 640
		},
		new()
		{
			Name = "yolo11n800fp32.onnx",
			Resolution = 800
		},
	];

	private static readonly IReadOnlyCollection<DetectedObjectExpectation> BusImageObjectExpectations =
	[
		new("person", 2, 4), // 2 persons are clearly visible and 2 more are only partially
		new("bus", 1), // bus is huge object on the image and locates in the center
		new("stop sign", 0, 1) // the sign is barely noticeable
	];

	public static IReadOnlyCollection<ImageSetInfo> ImageSets =>
	[
		new("bus{0}.png", [160, 224, 320, 480, 640, 800])
	];

	public static IEnumerable<DetectionTestData> MatchingSizesExpectations => Models.Join(
		ImageSets.SelectMany(set => set.Images),
		modelInfo => modelInfo.Resolution,
		imageInfo => imageInfo.Resolution,
		(model, image) => new DetectionTestData(model.Name, image.Name, BusImageObjectExpectations));

	public static IEnumerable<DetectionTestData> HigherInputSizesExpectations =>
		Models.SelectMany(info => SelectImagesWithHigherResolution(info, BusImageObjectExpectations));

	public static IEnumerable<DetectionTestData> SelectImagesWithHigherResolution(
		ModelInfo model,
		IReadOnlyCollection<DetectedObjectExpectation> objectsExpectations)
	{
		return ImageSets
			.SelectMany(set => set.Images)
			.Where(image => image.Resolution > model.Resolution)
			.Select(image => new DetectionTestData(model, image, objectsExpectations));
	}
}