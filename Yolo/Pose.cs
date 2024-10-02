namespace Yolo;

public readonly struct Pose
{
	public Detection Detection { get; }
	public IReadOnlyList<KeyPoint> KeyPoints { get; }

	public Pose(Detection detection, IReadOnlyList<KeyPoint> keyPoints)
	{
		Detection = detection;
		KeyPoints = keyPoints;
	}
}