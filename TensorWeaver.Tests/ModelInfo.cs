namespace TensorWeaver.Tests;

public sealed class ModelInfo
{
	public required string FileName { get; init; }
	public required IReadOnlyList<string> ClassesNames { get; init; }
	public string WebUrl { get; init; } = string.Empty;

	public async Task<byte[]> ReadDataAsync(CancellationToken cancellationToken)
	{
		using MemoryStream memoryStream = new();
		var readStream = await OpenReadAsync(cancellationToken);
		await readStream.CopyToAsync(memoryStream, cancellationToken);
		return memoryStream.ToArray();
	}

	private string FilePath => Path.Combine("Models", FileName);

	private async Task<Stream> OpenReadAsync(CancellationToken cancellationToken)
	{
		if (!File.Exists(FilePath) && !string.IsNullOrEmpty(WebUrl))
			await DownloadAsync(cancellationToken);
		return File.OpenRead(FilePath);
	}

	private async Task DownloadAsync(CancellationToken cancellationToken)
	{
		using var client = new HttpClient();
		var response = await client.GetAsync(WebUrl, cancellationToken);
		await using var targetStream = File.OpenWrite(FilePath);
		await response.Content.CopyToAsync(targetStream, cancellationToken);
	}
}