namespace TensorWeaver.Tests.Data;

public static class PoseTestsData
{
	public static IReadOnlyCollection<ModelInfo> Models { get; } =
	[
		new("yolov8n-pose-uint8.onnx", 640),
		new("yolov8n-pose160fp32.onnx", 160),
		new("yolov8n-pose224fp32.onnx", 224),
		new("yolov8n-pose320fp32.onnx", 320),
		new("yolov8n-pose480fp32.onnx", 480),
		new("yolov8n-pose640fp32.onnx", 640),
		new("yolov8n-pose800fp32.onnx", 800),
		new("yolo11n-pose160fp32.onnx", 160),
		new("yolo11n-pose224fp32.onnx", 224),
		new("yolo11n-pose320fp32.onnx", 320),
		new("yolo11n-pose480fp32.onnx", 480),
		new("yolo11n-pose640fp32.onnx", 640),
		new("yolo11n-pose800fp32.onnx", 800)
	];

	// the model doesn't know about bus and stop sign, so it is excluded
	private static readonly IReadOnlyCollection<DetectedObjectExpectation> BusImageObjectExpectations =
	[
		new("person", 2, 4) // 2 persons are clearly visible and 2 more are only partially.
	];

	public static IEnumerable<DetectionTestData> MatchingSizesExpectations => Models.Join(
		DetectionTestsData.ImageSets.SelectMany(set => set.Images),
		modelInfo => modelInfo.Resolution,
		imageInfo => imageInfo.Resolution,
		(model, image) => new DetectionTestData(model.Name, image.Name, BusImageObjectExpectations));

	public static IEnumerable<DetectionTestData> HigherInputSizesExpectations =>
		Models.SelectMany(model => DetectionTestsData.SelectImagesWithHigherResolution(model, BusImageObjectExpectations));
}