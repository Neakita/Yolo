namespace Yolo.OutputData;

public readonly struct Pose
{
	public Detection Detection { get; }
	public Classification Classification => Detection.Classification;
	public IReadOnlyList<KeyPoint> KeyPoints { get; }

	public Pose(Detection detection, IReadOnlyList<KeyPoint> keyPoints)
	{
		Detection = detection;
		KeyPoints = keyPoints;
	}
}