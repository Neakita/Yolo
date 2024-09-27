namespace Yolo;

public readonly struct Bounding
{
	public static Bounding Empty { get; } = new();

	public float Left { get; }
	public float Top { get; }
	public float Right { get; }
	public float Bottom { get; }
	public float Width => Right - Left;
	public float Height => Bottom - Top;

	public Bounding(float left, float top, float right, float bottom)
	{
		Left = left;
		Top = top;
		Right = right;
		Bottom = bottom;
	}
}