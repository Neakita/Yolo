namespace Yolo;

public struct KeyPoint3D
{
	public Vector2<float> Position { get; }
	public float Confidence { get; }

	public KeyPoint3D(Vector2<float> position, float confidence)
	{
		Position = position;
		Confidence = confidence;
	}
}