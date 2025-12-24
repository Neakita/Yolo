namespace TensorWeaver.Tests.Inference.Classificators;

internal static class ClassificatorModels
{
	private static readonly IReadOnlyList<string> ClassesNames = File.ReadAllLines("names.txt");

	public static ModelInfo YoloV8Nano => new()
	{
		FileName = "yolov8n224fp32cls.onnx",
		ClassesNames = ClassesNames
	};

	public static ModelInfo YoloV8NanoUInt8 => new()
	{
		FileName = "yolov8n-cls-uint8.onnx",
		ClassesNames = ClassesNames
	};
}