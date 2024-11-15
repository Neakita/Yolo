namespace Yolo.Tests.Data;

public static class ClassificationTestsData
{
	public static IEnumerable<string> Models =>
	[
		"yolov8n-cls-uint8.onnx",
		"yolov8n224fp32cls.onnx",
		"yolo11n224fp32cls.onnx",
	];

	public static IEnumerable<ImageClassificationExpectation> Expectations =>
	[
		new("pizza224.png", "pizza"),
		new("toaster224.png", "toaster")
	];
}