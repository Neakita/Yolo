using CommunityToolkit.Diagnostics;

namespace TensorWeaver.OutputData;

public readonly struct Bounding
{
	public static Bounding Empty { get; } = new();

	public static Bounding operator *(Bounding bounding, Vector2D<int> vector) => new(
		bounding.Left * vector.X,
		bounding.Top * vector.Y,
		bounding.Right * vector.X,
		bounding.Bottom * vector.Y);

	public static Bounding operator /(Bounding bounding, Vector2D<int> vector) => new(
		bounding.Left / vector.X,
		bounding.Top / vector.Y,
		bounding.Right / vector.X,
		bounding.Bottom / vector.Y);

	public float Left { get; }
	public float Top { get; }
	public float Right { get; }
	public float Bottom { get; }
	public float Width => Right - Left;
	public float Height => Bottom - Top;
	public float Area => Width * Height;

	public Bounding(float left, float top, float right, float bottom)
	{
		Left = left;
		Top = top;
		Right = right;
		Bottom = bottom;
		Guard.IsGreaterThanOrEqualTo(Width, 0);
		Guard.IsGreaterThanOrEqualTo(Height, 0);
	}

	public override string ToString()
	{
		return $"{nameof(Left)}: {Left}, {nameof(Top)}: {Top}, {nameof(Right)}: {Right}, {nameof(Bottom)}: {Bottom}";
	}
}