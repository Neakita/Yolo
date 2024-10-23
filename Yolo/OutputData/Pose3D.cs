namespace Yolo.OutputData;

public readonly struct Pose3D
{
	public Detection Detection { get; }
	public IReadOnlyList<KeyPoint3D> KeyPoints { get; }

	public Pose3D(Detection detection, IReadOnlyList<KeyPoint3D> keyPoints)
	{
		Detection = detection;
		KeyPoints = keyPoints;
	}
}