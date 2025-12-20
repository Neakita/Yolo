namespace TensorWeaver.Benchmark.Models;

public static class DetectionModels
{
	public static IEnumerable<ModelInfo> V8 => Directory.GetFiles(V8ModelsDirectoryPath).Select(CreateModelInfo);

	private static ModelInfo CreateModelInfo(string filePath)
	{
		return new ModelInfo
		{
			FilePath = filePath
		};
	}

	private static string V8ModelsDirectoryPath => Path.Combine("Models", "YoloV8");
}