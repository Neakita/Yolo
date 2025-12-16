namespace TensorWeaver.Benchmark.Models;

public static class ClassificationModels
{
	public static IEnumerable<ModelInfo> All => Directory.GetFiles(ModelsDirectoryPath).Select(CreateModelInfo);

	private static ModelInfo CreateModelInfo(string filePath)
	{
		return new ModelInfo
		{
			FilePath = filePath
		};
	}

	private static string ModelsDirectoryPath => Path.Combine("Models", "Classification");
}