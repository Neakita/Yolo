namespace TensorWeaver.Tests.Inference.Posers;

internal static class PoserModels
{
	private static IReadOnlyList<string> Names => ["person"];

	public static ModelInfo YoloV8Nano => new()
	{
		FileName = "yolov8n-pose800fp32.onnx",
		ClassesNames = Names
	};

	public static ModelInfo YoloV8NanoUInt8 => new()
	{
		FileName = "yolov8n-pose-uint8.onnx",
		ClassesNames = Names
	};
}