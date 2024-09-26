namespace Yolo;

public readonly struct Bounding
{
	public float Left { get; }
	public float Top { get; }
	public float Right { get; }
	public float Bottom { get; }

	public Bounding(float left, float top, float right, float bottom)
	{
		Left = left;
		Top = top;
		Right = right;
		Bottom = bottom;
	}
}