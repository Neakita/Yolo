namespace TensorWeaver.Tests;

internal static class Models
{
	private static readonly IReadOnlyList<string> COCONames = File.ReadAllLines("coco-names.txt");
	private static readonly IReadOnlyList<string> YoloNames = File.ReadAllLines("yolo-names.txt");

	public static readonly ModelInfo RFDETRNano = new()
	{
		FileName = "rf-detr-nano.onnx",
		ClassesNames = COCONames,
		WebUrl = "https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx"
	};

	public static readonly ModelInfo YoloV8NanoUInt8 = new()
	{
		FileName = "yolov8n-uint8.onnx",
		ClassesNames = YoloNames
	};

	public static readonly ModelInfo YoloV8Nano = new()
	{
		FileName = "yolov8n800fp32.onnx",
		ClassesNames = YoloNames
	};

	public static readonly ModelInfo YoloV10Nano = new()
	{
		FileName = "yolov10n800fp32.onnx",
		ClassesNames = YoloNames
	};

	public static readonly ModelInfo YoloV11Nano = new()
	{
		FileName = "yolo11n800fp32.onnx",
		ClassesNames = YoloNames
	};
}