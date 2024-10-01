namespace Yolo;

public readonly struct KeyPoint
{
	public Vector2<float> Position { get; }

	public KeyPoint(Vector2<float> position)
	{
		Position = position;
	}
}