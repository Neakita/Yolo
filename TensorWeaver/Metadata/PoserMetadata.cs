using Microsoft.ML.OnnxRuntime;

namespace TensorWeaver.Metadata;

public sealed class PoserMetadata
{
	public byte KeyPointsCount => _keyPointsCount;
	public byte KeyPointsDimensions => _keyPointsDimensions;

	public PoserMetadata(InferenceSession session)
	{
		ParseKeyPointsMetadata(session, out _keyPointsCount, out _keyPointsDimensions);
	}

	private readonly byte _keyPointsCount;
	private readonly byte _keyPointsDimensions;

	private static void ParseKeyPointsMetadata(InferenceSession session, out byte keyPointsCount, out byte keyPointsDimensions)
	{
		var shape = session.ModelMetadata.CustomMetadataMap["kpt_shape"];
		// parse '[17, 3]'
		shape = shape[1..^1];
		var split = shape.Split(", ");
		keyPointsCount = byte.Parse(split[0]);
		keyPointsDimensions = byte.Parse(split[1]);
	}
}