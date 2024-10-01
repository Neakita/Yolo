using System.Collections.Immutable;

namespace Yolo;

public readonly struct Pose3D
{
	public Detection Detection { get; }
	public ImmutableArray<KeyPoint3D> KeyPoints { get; }

	public Pose3D(Detection detection, ImmutableArray<KeyPoint3D> keyPoints)
	{
		Detection = detection;
		KeyPoints = keyPoints;
	}
}