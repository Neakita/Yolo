namespace TensorWeaver.Benchmark;

public sealed class ModelInfo
{
	public required string FilePath { get; init; }
	public string WebUrl { get; init; } = string.Empty;

	public byte[] ReadData()
	{
		using MemoryStream memoryStream = new();
		var readStream = OpenRead();
		readStream.CopyTo(memoryStream);
		return memoryStream.ToArray();
	}

	public override string ToString()
	{
		return Path.GetFileNameWithoutExtension(FilePath);
	}

	private Stream OpenRead()
	{
		if (!File.Exists(FilePath) && !string.IsNullOrEmpty(WebUrl))
			Download();
		return File.OpenRead(FilePath);
	}

	private void Download()
	{
		using var client = new HttpClient();
		var response = client.GetAsync(WebUrl).GetAwaiter().GetResult();
		using var targetStream = File.OpenWrite(FilePath);
		response.Content.CopyToAsync(targetStream);
	}
}