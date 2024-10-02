namespace Yolo;

public readonly struct KeyPoint
{
	public Vector2D<float> Position { get; }

	public KeyPoint(Vector2D<float> position)
	{
		Position = position;
	}
}