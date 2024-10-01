using System.Collections.Immutable;

namespace Yolo;

public readonly struct Pose
{
	public Detection Detection { get; }
	public ImmutableArray<KeyPoint> KeyPoints { get; }

	public Pose(Detection detection, ImmutableArray<KeyPoint> keyPoints)
	{
		Detection = detection;
		KeyPoints = keyPoints;
	}
}