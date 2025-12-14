namespace TensorWeaver.Yolo;

public sealed class YoloPoserMetadata
{
	public static YoloPoserMetadata Parse(IReadOnlyDictionary<string, string> metadata)
	{
		var shape = metadata["kpt_shape"];
		// parse '[17, 3]'
		shape = shape[1..^1];
		var split = shape.Split(", ");
		var keyPointsCount = byte.Parse(split[0]);
		var keyPointsDimensions = byte.Parse(split[1]);
		return new YoloPoserMetadata
		{
			KeyPointsCount = keyPointsCount,
			KeyPointsDimensions = keyPointsDimensions
		};
	}

	public byte KeyPointsCount { get; init; }
	public byte KeyPointsDimensions { get; init; }
}